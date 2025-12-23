# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
import torchvision.transforms.functional as TF

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.na_resize import NaResize, DivisibleCrop, Rearrange
from torchvision.transforms import ToTensor,Normalize,Compose
import math
from PIL import Image, ImageOps

class DreamIDV:

    def __init__(
        self,
        config,
        checkpoint_dir,
        dreamidv_ckpt,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            dreamidv_ckpt (`str`):
                Path of DreamID-V dit checkpoint
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {dreamidv_ckpt}")
        self.model = WanModel(
                             model_type=config.model_type,
                             dim=config.dim, 
                             ffn_dim=config.ffn_dim,
                             freq_dim=config.freq_dim,
                             in_dim=config.in_dim,
                             num_heads=config.num_heads,
                             num_layers=config.num_layers,
                             window_size=config.window_size,
                             qk_norm=config.qk_norm,
                             cross_attn_norm=config.cross_attn_norm,
                             eps=config.eps)
    
        logging.info(f"loading ckpt.")
        state = torch.load(dreamidv_ckpt, map_location=self.device)
        logging.info(f"loading state dict.")
        missing_keys, unexpected_keys = self.model.load_state_dict(state, strict=False)
        logging.info(f"len missing_keys: {len(missing_keys)}")
        logging.info(f"len unexpected_keys: {len(unexpected_keys)}")
        print(missing_keys)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)


    def load_image_latent_ref_ip_video(self,paths: str, size, device, frame_num):
        # Load size.
        patch_size = self.patch_size
        vae_stride = self.vae_stride

        def is_image_or_video_by_extension(file_path):
            image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}
            
            ext = os.path.splitext(file_path)[1].lower()
            if ext in image_exts:
                return "image"
            elif ext in video_exts:
                return "video"
            else:
                return "unknown"

        # import pdb; pdb.set_trace()
        # Load image and video.
        ref_vae_latents = {
            "image": [],
            "video": [],
            "mask": [],
            'pose_embedding':[]
        }
        video_h = 0
        video_w = 0
        from decord import VideoReader
        for i, path in enumerate(paths):
            
            if is_image_or_video_by_extension(path) == "video" and i == 0:
                print('ori_path', path)
                vr = VideoReader(path)
                frames = []

                for idx in range(vr.__len__()):
                    frame = vr[idx].asnumpy()
                    frame = Image.fromarray(frame)
                    frames.append(frame)
                video_w = frames[0].size[0]
                video_h = frames[0].size[1]
            
                frames = frames[:frame_num]

                frames_num = (len(frames)-1)//4*4 +1
                frames = frames[:frames_num]

                video_transform=Compose(
                    [
                        NaResize(
                            resolution=math.sqrt(size[0] * size[1]), 
                            downsample_only=True,
                        ),
                        DivisibleCrop((vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])),
                        Normalize(0.5, 0.5),
                        Rearrange("t c h w -> c t h w"),
                    ]
                )
                
                video_frames = video_transform(frames)
                video_vae_latent = self.vae.encode([video_frames], device)[0]
            
                ref_vae_latents["video"].append(video_vae_latent)

                

            elif is_image_or_video_by_extension(path) == "video" and i == 1:
                print('mask_path', path)
                vr = VideoReader(path)
                frames = []
                for idx in range(vr.__len__()):
                    frame = vr[idx].asnumpy()
                    frame = Image.fromarray(frame)
                    frames.append(frame)
                video_w = frames[0].size[0]
                video_h = frames[0].size[1]
                
                frames = frames[:frame_num]

                frames_num = (len(frames)-1)//4*4 +1
                frames = frames[:frames_num]
                video_mask_transform=Compose(
                    [
                        NaResize(
                            resolution=math.sqrt(size[0] * size[1]), # 256*448, 480*832
                            downsample_only=True,
                        ),
                        DivisibleCrop((vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])),
                        # Normalize(0.5, 0.5),
                        Rearrange("t c h w -> c t h w"),
                    ]
                )
                
                video_frames = video_mask_transform(frames) 
                video_vae_latent = self.vae.encode([video_frames], device)[0]
                video_vae_latent = video_vae_latent 
                ref_vae_latents["mask"].append(video_vae_latent)

            elif is_image_or_video_by_extension(path) == "video" and i == 3:
                print('pose_path', path)
                vr = VideoReader(path)
                frames = []
                for idx in range(vr.__len__()):
                    frame = vr[idx].asnumpy()
                    frame = Image.fromarray(frame)
                    frames.append(frame)

                frames = frames[:frame_num]
                frames_num = (len(frames)-1)//4*4 +1
                frames = frames[:frames_num]
                video_mask_transform=Compose(
                    [
                        NaResize(
                            resolution=math.sqrt(size[0] * size[1]), 
                            downsample_only=True,   
                        ),
                        DivisibleCrop((vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])),
                        Rearrange("t c h w -> c t h w"),
                    ]
                )
                
                video_frames = video_mask_transform(frames) 
                ref_vae_latents["pose_embedding"].append(video_frames)

            elif is_image_or_video_by_extension(path) == "image" and i == 2:

                with Image.open(path) as img:
                    img = img.convert("RGB")
                    img_ratio = img.width / img.height
                    target_ratio = video_w / video_h
                    
                    if img_ratio > target_ratio: 
                        new_width = video_w
                        new_height = int(new_width / img_ratio)
                    else:  
                        new_height = video_h
                        new_width = int(new_height * img_ratio)

                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    delta_w = video_w - img.size[0]
                    delta_h = video_h - img.size[1]
                    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                    new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))

                # Transform to tensor and normalize.
                image_transform=Compose(
                    [
                        NaResize(
                            resolution=math.sqrt(size[0] * size[1]), 
                            downsample_only=True,
                        ),
                        DivisibleCrop((vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])),
                        Normalize(0.5, 0.5),
                        Rearrange("t c h w -> c t h w"),
                    ]
                )
                new_img = image_transform([new_img])
                img_vae_latent = self.vae.encode([new_img], device)[0]
                ref_vae_latents["image"].append(img_vae_latent)
            else:
                print("Unknown file type.")
       
        ref_vae_latents["image"] = torch.cat(ref_vae_latents["image"], dim=0)
        ref_vae_latents["video"] = torch.cat(ref_vae_latents["video"], dim=0)
        ref_vae_latents["mask"] = torch.cat(ref_vae_latents["mask"], dim=0)
        ref_vae_latents["pose_embedding"] = torch.cat(ref_vae_latents["pose_embedding"], dim=0)
        
        return ref_vae_latents


        
    def generate(self,
                 input_prompt,
                 paths,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale_img=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
             paths ([`str`])`:
                Reference paths for swap face.
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        device = self.device   
        dtype = self.param_dtype
        
        latents_ref = self.load_image_latent_ref_ip_video(paths, 
                                                          size, 
                                                          device,
                                                          frame_num,
                                                        )

        latents_ref_video = latents_ref["video"].to(device,dtype)

        latents_ref_image = latents_ref["image"].to(device,dtype)
        pose_embedding = latents_ref["pose_embedding"].to(device,dtype) 
   
        pose_embedding = pose_embedding.unsqueeze(0)
        
        msk = latents_ref["mask"].to(device,dtype)

        F = frame_num
        target_shape = (self.vae.model.z_dim, latents_ref_video.shape[1],
                        latents_ref_video.shape[2],
                        latents_ref_video.shape[3])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
        
        y_i_v = latents_ref_video 
        img_ref = latents_ref_image

        arg_tiv = {
            'context': context, 
            'seq_len': seq_len,
            'y': [torch.concat([y_i_v, msk])],
            'pose_embedding': pose_embedding,
            'img_ref': [img_ref]
            }

        
        arg_tv = {
            'context': context,
            'seq_len': seq_len, 
            'y': [torch.concat([y_i_v, msk])],
            'pose_embedding': pose_embedding,
            'img_ref': [torch.zeros_like(img_ref)]
            }


        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            for _, t in enumerate(tqdm(timesteps)):
                
                timestep = [t]
                timestep = torch.stack(timestep)

                self.model.to(self.device)
                pos_tiv = self.model(latents, t=timestep, **arg_tiv)[0]
                pos_tv = self.model(latents, t=timestep, **arg_tv)[0]
                
                noise_pred = pos_tiv
                noise_pred += guide_scale_img * (pos_tiv - pos_tv) 
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents

            if offload_model:
                self.model.cpu()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
