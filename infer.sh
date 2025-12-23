# single-gpu
python generate_dreamidv.py \
    --size 832*480 \
    --ckpt_dir wan2.1-1.3B path \
    --dreamidv_ckpt dreamidv path  \
    --sample_steps 50 \
    --base_seed 42



# multi-gpu
torchrun --nproc_per_node=2 generate_dreamidv.py \
    --size 832*480 \
    --ckpt_dir wan2.1-1.3B path \
    --dreamidv_ckpt dreamidv path  \
    --sample_steps 50 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 2 \
    --ring_size 1 \
    --base_seed 42
