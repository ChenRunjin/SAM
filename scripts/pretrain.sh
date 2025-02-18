gpu=$1
echo CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nproc_per_node 1 main_pretrain.py \
    --model_config configs/llama_20m.json \
    --eval_every 1000 \
    --save_every 20000 \
    --dtype bfloat16 \
    --batch_size 128 \
    --total_batch_size 128 \
    --lr 0.01 \
    --warmup_steps 6000 \
    --num_training_steps 100000 \
    --optimizer adamw \
    --weight_decay 0 \
    --project SAM \
    --name adamw_20m_bs128 \
    --save_dir checkpoints/adamw_20m_bs128 \
    --restore_optimizer \
    --rho 0.05 \
    --single_gpu  