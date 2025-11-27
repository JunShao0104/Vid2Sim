import os
import logging
from argparse import ArgumentParser
import shutil

parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--mask_path", "-m", type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--glomap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()

colmap_command = '"{}"'.format(args.colmap_executable) if args.colmap_executable else "colmap"
glomap_command = '"{}"'.format(args.glomap_executable) if args.glomap_executable else "glomap"
magick_command = '"{}"'.format(args.magick_executable) if args.magick_executable else "magick"
use_gpu = 0 if args.no_gpu else 1

# Ensure images folder exists
if not os.path.exists(os.path.join(args.source_path, "images")):
    raise FileNotFoundError(f"No images folder at {args.source_path}/images")

if not args.skip_matching:
    os.makedirs(os.path.join(args.source_path, "distorted/sparse"), exist_ok=True)

    # ---------------------------
    # Feature extraction
    # ---------------------------
    feat_extract_cmd = (
        f'{colmap_command} feature_extractor '
        f'--database_path {args.source_path}/distorted/database.db '
        f'--image_path {args.source_path}/images '
        f'--ImageReader.single_camera 1 '
        f'--ImageReader.camera_model {args.camera} '
        f'--SiftExtraction.use_gpu {use_gpu}'
    )

    # if args.mask_path:
    #     if os.path.exists(args.mask_path):
    #         print(f"Using mask path: {args.mask_path}")
    #         feat_extract_cmd += f" --ImageReader.mask_path {args.mask_path}"
    # else:
    #     mask_folder = os.path.join(args.source_path, "masks")
    #     if os.path.exists(mask_folder):
    #         print(f"Using default mask path: {mask_folder}")
    #         feat_extract_cmd += f" --ImageReader.mask_path {mask_folder}"
    
    # Only add mask path if explicitly provided AND exists
    # if args.mask_path and os.path.exists(args.mask_path):
    #     print(f"Using mask path: {args.mask_path}")
    #     feat_extract_cmd += f" --ImageReader.mask_path {args.mask_path}"
    # else:
    #     print("No mask path provided or mask folder does not exist — ignoring masks.")

    print(f"Executing: {feat_extract_cmd}")
    exit_code = os.system(feat_extract_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # ---------------------------
    # Feature matching
    # ---------------------------
    feat_match_cmd = (
        f'{colmap_command} sequential_matcher '
        f'--database_path {args.source_path}/distorted/database.db '
        f'--SiftMatching.use_gpu {use_gpu} '
        f'--SiftMatching.max_num_matches 16384'
    )
    exit_code = os.system(feat_match_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # ---------------------------
    # Bundle adjustment / mapping
    # ---------------------------
    mapper_cmd = (
        f'{glomap_command} mapper '
        f'--database_path {args.source_path}/distorted/database.db '
        f'--image_path {args.source_path}/images '
        f'--output_path {args.source_path}/distorted/sparse'
    )
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

# ---------------------------
# Image undistortion
# ---------------------------
# Automatically detect sparse folder
sparse_folders = sorted(os.listdir(os.path.join(args.source_path, "distorted/sparse")))
if not sparse_folders:
    raise FileNotFoundError("No sparse folders found after mapping!")

input_sparse_path = os.path.join(args.source_path, "distorted/sparse", sparse_folders[0])

img_undist_cmd = (
    f'{colmap_command} image_undistorter '
    f'--image_path {args.source_path}/images '
    f'--input_path {input_sparse_path} '
    f'--output_path {args.source_path}/undistorted '
    f'--output_type COLMAP'
)
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Image undistortion failed with code {exit_code}. Exiting.")
    exit(exit_code)

# ---------------------------
# Move sparse files
# ---------------------------
# sparse_path = os.path.join(args.source_path, "sparse")
# os.makedirs(os.path.join(sparse_path, "0"), exist_ok=True)
# for file in sorted(os.listdir(sparse_path)):
#     if file == "0":
#         continue
#     shutil.move(os.path.join(sparse_path, file), os.path.join(sparse_path, "0", file))

# ---------------------------
# Move sparse files from distorted/sparse to sparse/0
# ---------------------------
src_sparse_path = os.path.join(args.source_path, "undistorted", "sparse")
dst_sparse_path = os.path.join(args.source_path, "sparse", "0")

os.makedirs(dst_sparse_path, exist_ok=True)

# The mapper usually creates a folder named "0"
# model_folder = sorted(os.listdir(src_sparse_path))[0]
# model_path = os.path.join(src_sparse_path, model_folder)
model_path = src_sparse_path

# Move each file into sparse/0
for file in os.listdir(model_path):
    shutil.copy2(os.path.join(model_path, file), os.path.join(dst_sparse_path, file))

# ---------------------------
# Optional resizing
# ---------------------------
if args.resize:
    print("Copying and resizing images...")
    for scale, folder in zip([50, 25, 12.5], ["images_2", "images_4", "images_8"]):
        os.makedirs(os.path.join(args.source_path, folder), exist_ok=True)

    files = sorted(os.listdir(os.path.join(args.source_path, "images")))
    for file in files:
        src_file = os.path.join(args.source_path, "images", file)
        for scale, folder in zip([50, 25, 12.5], ["images_2", "images_4", "images_8"]):
            dest_file = os.path.join(args.source_path, folder, file)
            shutil.copy2(src_file, dest_file)
            resize_cmd = f'{magick_command} mogrify -resize {scale}% "{dest_file}"'
            exit_code = os.system(resize_cmd)
            if exit_code != 0:
                logging.error(f"{scale}% resize failed for {dest_file}. Exiting.")
                exit(exit_code)

print(f"Finished processing {args.source_path}")
