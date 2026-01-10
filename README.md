# DreamID-V: Bridging the Image-to-Video Gap for High-Fidelity Face Swapping via Diffusion Transformer

<p align="center">
  <a href="https://guoxu1233.github.io/DreamID-V/">🌐 Project Page</a> |
  <a href="https://arxiv.org/abs/2601.01425">📜 Arxiv</a> |
  <a href="https://huggingface.co/XuGuo699/DreamID-V">🤗 Models</a> |
</p>

> **DreamID-V: Bridging the Image-to-Video Gap for High-Fidelity Face Swapping via Diffusion Transformer**<br>
> [Xu Guo](https://github.com/Guoxu1233/)<sup> * </sup>, [Fulong Ye](https://scholar.google.com/citations?user=-BbQ5VgAAAAJ&hl=zh-CN/)<sup> * </sup>, [Xinghui Li](https://crayon-shinchan.github.io/xinghui99.github.io/)<sup> *</sup>, [Pengqi Tu](https://openreview.net/profile?id=%7EPengqi_Tu1), [Pengze Zhang](https://pangzecheung.github.io/Homepage/), [Qichao Sun](https://github.com/sun631998316), [Songtao Zhao](https://openreview.net/profile?id=~Songtao_Zhao1)<sup> &dagger;</sup>, [Xiangwang Hou](https://scholar.google.com/citations?user=bpskf9kAAAAJ&hl=zh-CN)<sup> &dagger;</sup> [Qian He](https://scholar.google.com/citations?user=9rWWCgUAAAAJ)
> <br><sup> * </sup>Equal contribution,<sup> &dagger; </sup>Corresponding author
> <br>Tsinghua University | Intelligent Creation Team, ByteDance<br>

<p align="center">
<img src="assets/teaser.png" width=95%>
<p>

## 🔥 News
- [01/08/2026] 🔥 Thanks HM-RunningHub for supporting [ComfyUI](https://github.com/HM-RunningHub/ComfyUI_RH_DreamID-V) and Thanks Goldlionren for supporting the [low-VRAM version](https://github.com/Goldlionren/ComfyUI_JR_DreamID-V).
- [01/06/2026] 🔥 Our [paper](https://arxiv.org/abs/2601.01425) is released! 
- [01/05/2026] 🔥 Our code is released!
- [12/17/2025] 🔥 Our [project](https://guoxu1233.github.io/DreamID-V/) is released!
- [08/11/2025] 🎉 Our image version [DreamID](https://superhero-7.github.io/DreamID/) is accepted by SIGGRAPH Asia 2025!


## 💡 Usage Tips
- **Reference Image Preparation**: Please upload **cropped face images** (recommended resolution: 512x512) as reference. Avoid using full-body photos to ensure optimal identity preservation.
- **Inference Steps**: For simple scenes, you can reduce the sampling steps to **20** to significantly decrease inference time. 
  > *Note*: Our internal model based on Seedance1.0 achieves high quality in under 8 steps. Feel free to experience it at [CapCut](https://www.capcut.cn/).
- **Best Quality**: For the highest fidelity results, we recommend using a resolution of **1280x720**.
- **Enhanced Pose Detection**: We have resolved the previous pose detection issue by introducing [**DreamID-V-Wan-1.3B-DWPose**](https://github.com/bytedance/DreamID-V/tree/main?tab=readme-ov-file#dreamid-v-wan-13b-dwpose). This significantly improves stability and robustness in pose extraction.

## ⚡️ Quickstart

### Model Preparation
| Models       | Download Link                                                                                                                                           |    Notes                      |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| DreamID-V | 🤗 [Huggingface](https://huggingface.co/XuGuo699/DreamID-V)   | Supports 480P & 720P 
| Wan-2.1 | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | VAE & Text encoder

### Installation


Install dependencies:
```sh
# Ensure torch >= 2.4.0
pip install -r requirements.txt
```
#### DreamID-V-Wan-1.3B-DWPose
Please ensure the pose estimation models are placed in the correct directory as follows:
```text
DreamID-V/
└── pose/
    └── models/
        ├── dw-ll_ucoco_384.onnx 
        └── yolox_l.onnx         
```
- Single-GPU inference

``` sh
python generate_dreamidv_dwpose.py \
    --size 832*480 \
    --ckpt_dir wan2.1-1.3B path \
    --dreamidv_ckpt dreamidv.pth path  \
    --sample_steps 20 \
    --base_seed 42
```
- Multi-GPU inference using FSDP + xDiT USP

``` sh
pip install "xfuser>=0.4.1"
torchrun --nproc_per_node=2 generate_dreamidv_dwpose.py \
    --size 832*480 \
    --ckpt_dir wan2.1-1.3B path \
    --dreamidv_ckpt dreamidv.pth path  \
    --sample_steps 20 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 2 \
    --ring_size 1 \
    --base_seed 42
```

#### DreamID-V-Wan-1.3B-MediaPipe

- Single-GPU inference

``` sh
python generate_dreamidv.py \
    --size 832*480 \
    --ckpt_dir wan2.1-1.3B path \
    --dreamidv_ckpt dreamidv.pth path  \
    --sample_steps 20 \
    --base_seed 42
```

- Multi-GPU inference using FSDP + xDiT USP

``` sh
pip install "xfuser>=0.4.1"
torchrun --nproc_per_node=2 generate_dreamidv.py \
    --size 832*480 \
    --ckpt_dir wan2.1-1.3B path \
    --dreamidv_ckpt dreamidv.pth path  \
    --sample_steps 20 \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 2 \
    --ring_size 1 \
    --base_seed 42
```


## 👍 Acknowledgements
Our work builds upon and is greatly inspired by several outstanding open-source projects, including [Wan2.1](https://github.com/Wan-Video/Wan2.1), [Phantom](https://github.com/Phantom-video/Phantom), [OpenHumanVid](https://github.com/fudan-generative-vision/OpenHumanVid), [Follow-Your-Emoji](https://github.com/mayuelala/FollowYourEmoji), [DWPose](https://github.com/IDEA-Research/DWPose). We sincerely thank the authors and contributors of these projects for generously sharing their excellent codes and ideas.


## 📧 Contact
If you have any comments or questions regarding this open-source project, please open a new issue or contact [Xu Guo](https://github.com/Guoxu1233/) and [Fulong Ye](https://github.com/superhero-7).

## ⚠️ Ethics Statement
This project, **DreamID-V**, is intended for **academic research and technical demonstration purposes only**.
- **Prohibited Use**: Users are strictly prohibited from using this codebase to generate content that is illegal, defamatory, pornographic, harmful, or infringes upon the privacy and rights of others.
- **Responsibility**: Users bear full responsibility for the content they generate. The authors and contributors of this project assume no liability for any misuse or consequences arising from the use of this software.
- **AI Labeling**: We strongly recommend marking generated videos as "AI-Generated" to prevent misinformation.
By using this software, you agree to adhere to these guidelines and applicable local laws.

## ⭐ Citation

If you find our work helpful, please consider citing our paper and leaving valuable stars

```
@misc{guo2026dreamidvbridgingimagetovideogaphighfidelity,
      title={DreamID-V:Bridging the Image-to-Video Gap for High-Fidelity Face Swapping via Diffusion Transformer}, 
      author={Xu Guo and Fulong Ye and Xinghui Li and Pengqi Tu and Pengze Zhang and Qichao Sun and Songtao Zhao and Xiangwang Hou and Qian He},
      year={2026},
      eprint={2601.01425},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.01425}, 
}
```
