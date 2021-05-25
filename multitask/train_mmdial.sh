export MODEL=$3
export MODEL_NAME=$4
export BATCH=$5
export OUTPUT=output_mmdial/${MODEL_NAME} 
export TRAIN_FILE=./resources/train_simpletod
export TEST_FILE=./resources/val_simpletod


CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --nproc_per_node=$2  main.py \
    --output_dir=$OUTPUT \
    --model_type=$MODEL \
    --model_name_or_path=$MODEL_NAME \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --evaluate_during_training \
    --save_steps 2000 \
    --logging_steps 4000 \
    --per_gpu_train_batch_size $BATCH \
    --num_train_epochs 50 \
    --save_total_limit $6 \
    