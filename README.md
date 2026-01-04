# DreamID-V: Bridging the Image-to-Video Gap for High-Fidelity Face Swapping via Diffusion Transformer

<p align="center">
  <a href="https://guoxu1233.github.io/DreamID-V/">🌐 Project Page</a> |
  <a href="https://arxiv.org5">📜 Arxiv</a> |
  <a href="https://huggingface.co">🤗 Models</a> |
</p>

> **DreamID-V: Bridging the Image-to-Video Gap for High-Fidelity Face Swapping via Diffusion Transformer**<br>
> [Xu Guo](https://github.com/Guoxu1233/)<sup> * </sup>, [Fulong Ye](https://scholar.google.com/citations?user=-BbQ5VgAAAAJ&hl=zh-CN/)<sup> * </sup>, [Xinghui Li](https://crayon-shinchan.github.io/xinghui99.github.io/)<sup> *</sup>, [Pengqi Tu](https://openreview.net/profile?id=%7EPengqi_Tu1), [Pengze Zhang](https://pangzecheung.github.io/Homepage/), [Qichao Sun](https://github.com/sun631998316), [Songtao Zhao](https://openreview.net/profile?id=~Songtao_Zhao1)<sup> &dagger;</sup>, [Xiangwang Hou](https://scholar.google.com/citations?user=bpskf9kAAAAJ&hl=zh-CN)<sup> &dagger;</sup> [Qian He](https://scholar.google.com/citations?user=9rWWCgUAAAAJ)
> <br><sup> * </sup>Equal contribution,<sup> &dagger; </sup>Corresponding author
> <br>Tsinghua University | Intelligent Creation Team, ByteDance<br>

<p align="center">
<img src="assets/teaser.png" width=95%>
<p>

## ⚡️ Quickstart

### Installation


Install dependencies:
```sh
# Ensure torch >= 2.4.0
pip install -r requirements.txt
```





#### DreamID-V-Wan-1.3B

- Single-GPU inference

``` sh
python generate_dreamidv.py \
    --size 832*480 \
    --ckpt_dir wan2.1-1.3B path \
    --dreamidv_ckpt dreamidv path  \
    --sample_steps 50 \
    --base_seed 42
```

- Multi-GPU inference using FSDP + xDiT USP

``` sh
pip install "xfuser>=0.4.1"
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
```

## 👍 Acknowledgements
Our work builds upon and is greatly inspired by several outstanding open-source projects, including [Wan2.1](https://github.com/Wan-Video/Wan2.1), [Phantom](https://github.com/Phantom-video/Phantom), [OpenHumanVid](https://github.com/fudan-generative-vision/OpenHumanVid), [Follow-Your-Emoji](https://github.com/mayuelala/FollowYourEmoji). We sincerely thank the authors and contributors of these projects for generously sharing their excellent codes and ideas.


## 📧 Contact
If you have any comments or questions regarding this open-source project, please open a new issue or contact [Xu Guo](https://github.com/Guoxu1233/).
