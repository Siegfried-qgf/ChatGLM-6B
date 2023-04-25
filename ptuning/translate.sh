DATASET=ultrachat
TIME=$(date '+%H:%M')

CUDA_VISIBLE_DEVICES=1 python3 translate.py \
    --do_predict \
    --test_file /workspace/ChatGLM-6B/dataset/ultrachat/train_0.json \
    --overwrite_cache \
    --prompt_column data \
    --response_column data \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir ./output/$DATASET/$TIME \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 2048 \
    --generation_max_length 2048\
    --per_device_eval_batch_size 4 \
    --max_predict_samples 100\
    --predict_with_generate \
    --source_prefix 请把下面这段话翻译成中文（流畅并且不能出现英文）:
    