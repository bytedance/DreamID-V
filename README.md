# DreamID-Omni: Unified Framework for Controllable Human-Centric Audio-Video Generation

<p align="center">
  <a href="https://guoxu1233.github.io/DreamID-Omni/">🌐 Project Page</a> |
  <a href="https://arxiv.org/abs/2602.12160">📜 Arxiv</a> |
  <a href="https://huggingface.co/XuGuo699/DreamID-Omni">🤗 Models</a> |
</p>

> **DreamID-Omni: Unified Framework for Controllable Human-Centric Audio-Video Generation**<br>
> [Xu Guo](https://github.com/Guoxu1233/)<sup> * </sup>, [Fulong Ye](https://scholar.google.com/citations?user=-BbQ5VgAAAAJ&hl=zh-CN/)<sup> * </sup>, [Qichao Sun](https://github.com/sun631998316)<sup> *</sup><sup>&dagger;</sup>, [Liyang Chen](https://leoniuschen.github.io/),  [Bingchuan Li](https://scholar.google.com/citations?user=ac5Se6QAAAAJ&hl=zh-CN)<sup> &dagger;</sup>, [Pengze Zhang](https://pangzecheung.github.io/Homepage/), [Jiawei Liu](https://scholar.google.com/citations?user=X21Fz-EAAAAJ&hl=zh-CN), [Songtao Zhao](https://openreview.net/profile?id=~Songtao_Zhao1)<sup> &sect;</sup>,  [Qian He](https://scholar.google.com/citations?user=9rWWCgUAAAAJ), [Xiangwang Hou](https://scholar.google.com/citations?user=bpskf9kAAAAJ&hl=zh-CN)<sup> &sect;</sup>
> <br><sup> * </sup>Equal contribution,<sup> &dagger; </sup>Project lead, <sup> &sect; </sup>Corresponding author 
> <br>Tsinghua University | Intelligent Creation Team, ByteDance<br>

<p align="center">
<img src="assets/teaser.png" width=95%>
<p>

## 🔥 News
- [03/13/2026] 🔥 Our v1 version [code](https://github.com/Guoxu1233/DreamID-Omni) for R2AV is released!
- [02/13/2026] 🔥 Our [paper](https://arxiv.org/abs/2602.12160) is released!
- [01/05/2026] 🔥 The code for our previous work, [DreamID-V](https://github.com/bytedance/DreamID-V), has been released!


## 🎬 Demo
<div align="center">
  <video src="https://github.com/user-attachments/assets/21dda629-aee0-4c9e-b1ae-09552880a336" width="70%" poster=""> </video>
</div>


## ⚡️ Quickstart
### Installation
```bash
python3 download_weights.py
conda create -n dreamid_omni python=3.11
conda activate dreamid_omni
pip install torch==2.6.0 torchvision torchaudio
pip install -r requirements.txt
pip install flash_attn --no-build-isolation
```
### Inference

#### Single-GPU inference
``` sh
python3 inference_r2av.py --config-file dreamid_omni/configs/inference/inference_r2av.yaml
```

#### Multi-GPU inference
``` sh
torchrun --nnodes 1 --nproc_per_node 8 inference_r2av.py --config-file dreamid_omni/configs/inference/inference_r2av.yaml
```
Before running multi-GPU inference, please open `dreamid_omni/configs/inference/inference_r2av.yaml` and set sp_size: 8



## 🎨 How to Create 
Our prompts use the following special tags to control characters and speech:
- **Subject Identity**: `<sub1>`, `<sub2>` - Represents the character IPs provided in your input images (e.g., `<img1>` corresponds to `<sub1>`). Use these tags in your prompt to specify who is acting or speaking.
- **Speech**: `<S>Your speech content here<E>` - Text enclosed in these tags will be converted to speech using the corresponding character's reference audio.
### 💡 Structure Example
We provide example prompts to help you get started with DreamID-Omni:
- **Single-person generation**: `test_case/oneip`
- **Multi-person generation**: `test_case/twoip`


## 🙏 Acknowledgements

Our work builds upon and is greatly inspired by several outstanding open-source projects, including [Ovi](https://github.com/character-ai/Ovi), [Wan2.2](https://github.com/Wan-Video/Wan2.2), [MMAudio](https://github.com/hkchengrex/MMAudio), [Phantom](https://github.com/Phantom-video/Phantom), [HuMo](https://github.com/Phantom-video/HuMo), [OpenHumanVid](https://github.com/fudan-generative-vision/OpenHumanVid). We sincerely thank the authors and contributors of these projects for generously sharing their excellent codes and ideas.


## 📧 Contact
If you have any comments or questions regarding this open-source project, please open a new issue or contact [Xu Guo](https://github.com/Guoxu1233/), [Fulong Ye](https://github.com/superhero-7) and [Qichao Sun](https://github.com/sun631998316).

## ⚠️ Ethics Statement
This project, **DreamID-Omni**, is intended for **academic research and technical demonstration purposes only**.
- **Prohibited Use**: Users are strictly prohibited from using this codebase to generate content that is illegal, defamatory, pornographic, harmful, or infringes upon the privacy and rights of others.
- **Responsibility**: Users bear full responsibility for the content they generate. The authors and contributors of this project assume no liability for any misuse or consequences arising from the use of this software.
- **AI Labeling**: We strongly recommend marking generated videos as "AI-Generated" to prevent misinformation.
By using this software, you agree to adhere to these guidelines and applicable local laws.

## ⭐ Citation

If you find our work helpful, please consider citing our paper and leaving valuable stars

```
@misc{guo2026dreamidomni,
      title={DreamID-Omni: Unified Framework for Controllable Human-Centric Audio-Video Generation}, 
      author={Xu Guo and Fulong Ye and Qichao Sun and Liyang Chen and Bingchuan Li and Pengze Zhang and Jiawei Liu and Songtao Zhao and Qian He and Xiangwang Hou},
      year={2026},
      eprint={2602.12160},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.12160}, 
}
```


