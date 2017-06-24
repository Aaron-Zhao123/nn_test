CHECKPOINT_DIR=/local/scratch/yaz21/NN_models/slim_models/checkpoints/mobilenet_v1_1.0_224.ckpt
DATASET_DIR=/local/scratch/ssd
python mobilenet_eval.py \
    --checkpoint_path=${CHECKPOINT_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=mobilenet_v1
