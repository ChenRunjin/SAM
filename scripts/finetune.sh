gpu=${1:-0}
port=$((RANDOM % (18888 - 7000 + 1) + 7000))
echo CUDA_VISIBLE_DEVICES=$gpu python main_finetune.py \
    --model_name microsoft/phi-2 \
    --dataset_name MATH_instruct \
    --batch_size 32 \
    --micro_batch_size 1 \
    --eval_batch_size 4 \
    --num_train_epochs 10 \
    --learning_rate 5e-6 \
    --output_dir ckpt_my/MATH_instruct/phi-2_function_sam_rho0.001_epoch10_lr5e-6_bs32_new  \
    --optimizer_name function_sam \
    --eval_steps 500 \
    --save_steps 500 \
    --sam_rho 0.001 \
    --cutoff_len 800
    # --gradient_accumulation_steps 16 \
    # --deepspeed ./configs/zero3.json \
    # --optimizer sam 