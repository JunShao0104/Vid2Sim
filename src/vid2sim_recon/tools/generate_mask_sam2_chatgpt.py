import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from grounding_sam2_utils.video_utils import create_video_from_images
from grounding_sam2_utils.common_utils import CommonUtils
from grounding_sam2_utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy

def setup_environment():
    """Sets up the computing environment, especially GPU settings."""
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    return device

def load_models(device, model_name, grounding_model_id):
    """Loads the SAM2 and Grounding DINO models."""
    image_predictor = SAM2ImagePredictor.from_pretrained(model_name)
    video_predictor = SAM2VideoPredictor.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)
    return video_predictor, image_predictor, processor, grounding_model

def prepare_directories_and_frames(video_dir, output_dir):
    """Creates output directories and returns sorted frame names."""
    CommonUtils.creat_dirs(output_dir)
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    result_dir = os.path.join(output_dir, "result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)
    CommonUtils.creat_dirs(result_dir)

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return mask_data_dir, json_data_dir, result_dir, frame_names

def save_frame_results(frame_masks_info, mask_data_dir, json_data_dir):
    """Saves the mask and JSON data for a single frame."""
    mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
    for obj_id, obj_info in frame_masks_info.labels.items():
        mask_img[obj_info.mask == True] = obj_id

    mask_img_np = mask_img.cpu().numpy().astype(np.uint16)
    mask_path = os.path.join(mask_data_dir, frame_masks_info.mask_name)
    np.save(mask_path, mask_img_np)

    json_data = frame_masks_info.to_dict()
    json_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
    with open(json_path, "w") as f:
        json.dump(json_data, f)

def process_frame_batch(start_frame_idx, frame_names, step, video_dir, text, device,
                          processor, grounding_model, image_predictor, video_predictor,
                          inference_state, global_masks, objects_count,
                          mask_data_dir, json_data_dir, prompt_type):
    """Processes a batch of frames: detection, SAM prediction, video propagation, saving results."""
    for idx in range(start_frame_idx, min(start_frame_idx + step, len(frame_names))):
        img_name = frame_names[idx]
        img_path = os.path.join(video_dir, img_name)
        image = Image.open(img_path)
        image_base_name = os.path.splitext(img_name)[0]

        current_frame_masks = MaskDictionaryModel(promote_type=prompt_type, mask_name=f"mask_{image_base_name}.npy")


        # Run Grounding DINO
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=0.25, text_threshold=0.25, target_sizes=[image.size[::-1]]
        )

        input_boxes = results[0]["boxes"]
        labels = results[0]["labels"]

        if input_boxes.shape[0] > 0:
            # Run SAM Image Predictor
            image_predictor.set_image(np.array(image.convert("RGB")))
            masks, scores, logits = image_predictor.predict(
                point_coords=None, point_labels=None, box=input_boxes, multimask_output=False,
            )

            if masks.ndim == 2: # Single mask
                masks = masks[None]
            elif masks.ndim == 4: # Batch
                masks = masks.squeeze(1)

            current_frame_masks.add_new_frame_annotation(
                mask_list=torch.tensor(masks).to(device),
                box_list=torch.tensor(input_boxes).to(device),
                label_list=labels
            )

            objects_count = current_frame_masks.update_masks(
                tracking_annotation_dict=global_masks, iou_threshold=0.8, objects_count=objects_count
            )

        else:
            # No object detected: propagate previous masks if available
            if global_masks.labels:
                current_frame_masks = copy.deepcopy(global_masks)

        # Save mask for this frame (even empty)
        if current_frame_masks.labels:
            first_mask = next(iter(current_frame_masks.labels.values())).mask
            current_frame_masks.mask_height = first_mask.shape[-2]
            current_frame_masks.mask_width = first_mask.shape[-1]
        else:
            # Use image dimensions for empty masks
            current_frame_masks.mask_height, current_frame_masks.mask_width = image.size[1], image.size[0]

        save_frame_results(current_frame_masks, mask_data_dir, json_data_dir)

        # Update global_masks for next frame
        if current_frame_masks.labels:
            global_masks = copy.deepcopy(current_frame_masks)

    return global_masks, objects_count

def generate_visualization(video_dir, mask_data_dir, json_data_dir, output_dir, result_dir, output_video_path, frame_rate=15):
    """Draws masks and boxes on frames and saves the final annotated video."""
    print("Generating visualization...")
    CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, output_dir, result_dir)
    create_video_from_images(result_dir, output_video_path, frame_rate=frame_rate)
    print(f"Output video saved to: {output_video_path}")

def main(model_name='facebook/sam2-hiera-large', 
        grounding_model_id='IDEA-Research/grounding-dino-base', 
        text_prompt='car.', 
        video_dir='notebooks/videos/car', 
        output_dir='./outputs', 
        result_dir='./results',
        step=20, 
        prompt_type='mask'):

    device = setup_environment()
    video_predictor, image_predictor, processor, grounding_model = load_models(device, model_name, grounding_model_id)
    mask_data_dir, json_data_dir, vis_dir, frame_names = prepare_directories_and_frames(video_dir, output_dir)

    inference_state = video_predictor.init_state(video_path=video_dir, offload_video_to_cpu=True, async_loading_frames=True)
    global_masks = MaskDictionaryModel(promote_type=prompt_type)
    objects_count = 0

    for start_frame_idx in range(0, len(frame_names), step):
        global_masks, objects_count = process_frame_batch(
            start_frame_idx, frame_names, step, video_dir, text_prompt, device,
            processor, grounding_model, image_predictor, video_predictor,
            inference_state, global_masks, objects_count,
            mask_data_dir, json_data_dir, prompt_type
        )

    output_video_path = os.path.join(output_dir, "output.mp4")
    generate_visualization(video_dir, mask_data_dir, json_data_dir, vis_dir, result_dir, output_video_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate dynamic masks using GroundingSAM2')
    parser.add_argument('--model_name', type=str, default='facebook/sam2-hiera-large')
    parser.add_argument('--grounding_model_id', type=str, default='IDEA-Research/grounding-dino-base')
    parser.add_argument('--text_prompt', type=str, default='car.')
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--result_dir', type=str, default='./results')
    args = parser.parse_args()

    main(
        model_name=args.model_name,
        grounding_model_id=args.grounding_model_id,
        text_prompt=args.text_prompt,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        result_dir=args.result_dir
    )
