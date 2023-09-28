# python scripts/run.py \
#     --config-name=arss01 \
#     data_path=/mnt/nas/share-all/caizebin/03.dataset/nerf/data/f2nerf \
#     dataset_name=20230828T164146+0800_Capture_OPPO_PDEM30_molly0828 \
#     mode=train \
#     +work_dir=$(pwd)

python scripts/run.py \
    --config-name=arss01 \
    data_path=/mnt/nas/share-all/caizebin/03.dataset/nerf/data/f2nerf \
    dataset_name=20230828T164146+0800_Capture_OPPO_PDEM30_molly0828 \
    mode=mesh_mata \
    +work_dir=$(pwd) \
    is_continue=true

# python scripts/run.py \
#     --config-name=arss01 \
#     data_path=/mnt/nas/share-all/caizebin/03.dataset/nerf/data/f2nerf \
#     dataset_name=20230828T164146+0800_Capture_OPPO_PDEM30_molly0828 \
#     mode=test \
#     +work_dir=$(pwd) \
#     is_continue=true
