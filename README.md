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


#### DreamID-V-Wan-1.3B

- Single-GPU inference

``` sh
python generate_dreamidv.py \
    --size 832*480 \
    --ckpt_dir wan2.1-1.3B path \
    --dreamidv_ckpt dreamidv.pth path  \
    --sample_steps 50 \
    --base_seed 42
```

- Multi-GPU inference using FSDP + xDiT USP

``` sh
pip install "xfuser>=0.4.1"
torchrun --nproc_per_node=2 generate_dreamidv.py \
    --size 832*480 \
    --ckpt_dir wan2.1-1.3B path \
    --dreamidv_ckpt dreamidv.pth path  \
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
