#single-gpu inference
python3 inference_r2av.py --config-file dreamid_omni/configs/inference/inference_r2av.yaml

#multi-gpu inference set sp_size=8 in inference_r2av.yaml
torchrun --nnodes 1 --nproc_per_node 8 inference_r2av.py --config-file dreamid_omni/configs/inference/inference_r2av.yaml