import os
import argparse
import logging
import time
from huggingface_hub import snapshot_download




# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def timed_download(repo_id: str, local_dir: str, allow_patterns: list):
    """Download files from HF repo and log time + destination."""
    logging.info(f"Starting download from {repo_id} into {local_dir}")
    start_time = time.time()

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )

    elapsed = time.time() - start_time
    logging.info(
        f"✅ Finished downloading {repo_id} "
        f"in {elapsed:.2f} seconds. Files saved at: {local_dir}"
    )

def main(output_dir: str):
    # Wan2.2
    wan_dir = os.path.join(output_dir, "Wan2.2-TI2V-5B")
    timed_download(
        repo_id="Wan-AI/Wan2.2-TI2V-5B",
        local_dir=wan_dir,
        allow_patterns=[
            "google/*",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.2_VAE.pth"
        ]
    )

    # MMAudio
    mm_audio_dir = os.path.join(output_dir, "MMAudio")
    timed_download(
        repo_id="hkchengrex/MMAudio",
        local_dir=mm_audio_dir,
        allow_patterns=[
            "ext_weights/best_netG.pt",
            "ext_weights/v1-16.pth"
        ]
    )

    dreamid_omni_dir = os.path.join(output_dir, "DreamI_Omni")

    timed_download(
        repo_id="XuGuo699/DreamID-Omni",
        local_dir=dreamid_omni_dir,
        allow_patterns=["dreamid_omni.safetensors"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Hugging Face")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ckpts",
        help="Base directory to save downloaded models"
    )

    args = parser.parse_args()
    main(args.output_dir)