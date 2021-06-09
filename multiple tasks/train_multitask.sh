export MODEL=$2
export MODEL_NAME=$3
export BATCH=$4
export OUTPUT=output/${MODEL_NAME}
export TRAIN_FILE=./resources/train.simpletod
export TEST_FILE=./resources/test.simpletod


CUDA_VISIBLE_DEVICES=$1 python main.py \
    --output_dir=$OUTPUT \
    --model_type=$MODEL \
    --model_name_or_path=$MODEL_NAME \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --evaluate_during_training \
    --save_steps 4170 \
    --logging_steps 66730 \
    --per_gpu_train_batch_size $BATCH \
    --num_train_epochs 80