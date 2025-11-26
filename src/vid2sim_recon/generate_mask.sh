# Heuristic dynamic masks category
MASK_PROMPT="person.pedestrian.cyclist.child.adult.bag.backpack.handbag.suitcase.hat.shoes.cloth.wheelchair"

# Generate dynamic masks from SAM2 (recommended)
python tools/generate_mask_sam2.py \
--text_prompt $MASK_PROMPT \
--video_dir $1/images \
--output_dir $1/tmp \
--result_dir $1/masks

# Generate dynamic masks from DEVA (deprecated)
# cd /scratch/lzhao360/Vid2Sim/submodules/vid2sim-deva-segmentation
# bash generate_mask.sh $1 $MASK_PROMPT