import os
import sys
import uuid
import cv2
import glob
import torch
import logging
import math
from textwrap import indent
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageOps
from dreamid_omni.distributed_comms.parallel_states import get_sequence_parallel_state, nccl_info
from dreamid_omni.utils.model_loading_utils import init_fusion_score_model_ovi, init_text_model, init_mmaudio_vae, init_wan_vae_2_2, load_fusion_checkpoint
from dreamid_omni.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from dreamid_omni.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
import traceback
from omegaconf import OmegaConf
from dreamid_omni.utils.processing_utils import clean_text, preprocess_image_tensor, snap_hw_to_multiple_of_32, scale_hw_to_area_divisible
import re
import librosa
from dreamid_omni.utils.divisible_crop import DivisibleCrop
from torchvision.transforms import ToTensor,Normalize,Compose
from dreamid_omni.utils.rearrange import Rearrange
from dreamid_omni.utils.resize import NaResize



DEFAULT_CONFIG = OmegaConf.load('dreamid_omni/configs/inference/inference_r2av.yaml')

NAME_TO_MODEL_SPECS_MAP = {
    "dreamid_omni": {
        "path": "dreamid_omni.safetensors",
        "video_latent_length": 31,
        "audio_latent_length": 157,
    }
}


class DreamIDOmniEngine:
    def __init__(self, config=DEFAULT_CONFIG, device=0, target_dtype=torch.bfloat16):
        # Load fusion model
        self.device = device
        self.target_dtype = target_dtype
        meta_init = True
        self.cpu_offload = config.get("cpu_offload", False)
        if self.cpu_offload:
            logging.info("CPU offloading is enabled. Initializing all models aside from VAEs on CPU")

        model, video_config, audio_config = init_fusion_score_model_ovi(rank=device, meta_init=meta_init)

     
        if not meta_init:
            model = (
                model.to(device=device if not self.cpu_offload else "cpu")
                .eval()
            )

        # Load VAEs
        vae_model_video = init_wan_vae_2_2(config.ckpt_dir, rank=device)
        vae_model_video.model.requires_grad_(False).eval()
        vae_model_video.model = vae_model_video.model.bfloat16()
        self.vae_model_video = vae_model_video

        vae_model_audio = init_mmaudio_vae(config.ckpt_dir, rank=device)
        vae_model_audio.requires_grad_(False).eval()
        self.vae_model_audio = vae_model_audio.bfloat16()

        # Load T5 text model
        self.text_model = init_text_model(config.ckpt_dir, rank=device, cpu_offload=self.cpu_offload)
        if config.get("shard_text_model", False):
            raise NotImplementedError("Sharding text model is not implemented yet.")
        if self.cpu_offload:
            self.offload_to_cpu(self.text_model.model)

        model_name = config.get("model_name", "dreamid_omni")
        self.model_name = model_name
        assert model_name in NAME_TO_MODEL_SPECS_MAP, f"Model name {model_name} not found in predefined model name to path map."
        model_specs = NAME_TO_MODEL_SPECS_MAP[model_name]
        basename = model_specs["path"]

        
        checkpoint_path = os.path.join(
            config.ckpt_dir,
            "DreamID_Omni",
            basename,
        )

        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"REQUIRED fusion checkpoint not found in {config.ckpt_dir}, please download...")

        load_fusion_checkpoint(model, checkpoint_path=checkpoint_path, from_meta=meta_init)

        if meta_init:
            model = model.to(device=device if not self.cpu_offload else "cpu").eval()
            model.set_rope_params()
        self.model = model

        self.image_model = None
        
        self.audio_latent_channel = audio_config.get("in_dim")
        self.video_latent_channel = video_config.get("in_dim")
        self.video_latent_length = model_specs["video_latent_length"]
        self.audio_latent_length = model_specs["audio_latent_length"]
    


    def load_image_latent_ref_ip_video(self, paths: str, video_frame_height_width, device):
        # Load size.
        patch_size = self.model.video_model.patch_size
        vae_stride = [4, 16, 16]


        def is_image_or_video_by_extension(file_path):
            image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}
            audio_exts = {".wav", ".mp3", ".aac", ".flac"}
            
            ext = os.path.splitext(file_path)[1].lower()
            if ext in image_exts:
                return "image"
            elif ext in video_exts:
                return "video"
            elif ext in audio_exts:
                return "audio"
            else:
                return "unknown"


        ref_vae_latents = {
            "image": [],
            "audio": [],
        }
        video_h = video_frame_height_width[0]
        video_w = video_frame_height_width[1]
        if self.cpu_offload:
             self.vae_model_video.model = self.vae_model_video.model.to(self.device)
        ref_audio_lengths = []
        for path in paths:
            if is_image_or_video_by_extension(path) == "image":
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

                image_transform=Compose(
                    [
                        NaResize(
                            resolution=math.sqrt(video_frame_height_width[0] * video_frame_height_width[1]), # 256*448, 480*832
                            mode="area",
                            downsample_only=True,
                        ),
                        DivisibleCrop((vae_stride[1] * patch_size[1], vae_stride[2] * patch_size[2])),
                        Normalize(0.5, 0.5),
                        Rearrange("t c h w -> c t h w"),
                    ]
                )
                new_img = image_transform([new_img])
                new_img = new_img.transpose(0, 1)
                new_img = new_img.to(self.device)
                new_img = new_img.to(self.target_dtype)

                with torch.no_grad():
                    img_vae_latent = self.vae_model_video.wrapped_encode(new_img[:, :, None]).to(self.target_dtype).squeeze(0)
                ref_vae_latents["image"].append(img_vae_latent)

            elif is_image_or_video_by_extension(path) == "audio":
                audio_array, sr = librosa.load(path, sr=16000)
                audio_array = audio_array[int(sr * 1): int(sr * 2)]

                audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)
                audio_vae_latent = self.vae_model_audio.wrapped_encode(audio_tensor)
                audio_length = audio_vae_latent.shape[2]
                ref_audio_lengths.append(audio_length)
                audio_vae_latent = audio_vae_latent.squeeze(0).transpose(0, 1)
                ref_vae_latents["audio"].append(audio_vae_latent)
            else:
                print("Unknown file type.")
        
        ref_vae_latents["image"] = torch.cat(ref_vae_latents["image"], dim=1)
        ref_vae_latents["audio"] = torch.cat(ref_vae_latents["audio"], dim=0)
    
        return ref_vae_latents, ref_audio_lengths

    
    @torch.inference_mode()
    def generate(self,
                    text_prompt, 
                    image0_path=None, 
                    image1_path=None, 
                    audio0_path=None,
                    audio1_path=None, 
                    video_frame_height_width=None,
                    seed=100,
                    solver_name="unipc",
                    sample_steps=50,
                    shift=5.0,
                    video_cfg_scale = 3.0,
                    video_ref_cfg_scale = 1.5,
                    audio_cfg_scale = 4.0,
                    audio_ref_cfg_scale = 2.0,
                    video_negative_prompt="",
                    audio_negative_prompt=""
                ):
        params = {
            "Mode": "Two-Person Reference",
            "Text Prompt": text_prompt,
            "Image0 Path": image0_path,
            "Image1 Path": image1_path,
            "Audio0 Path": audio0_path,
            "Audio1 Path": audio1_path,
            "Frame Height Width": video_frame_height_width,
            "Seed": seed,
            "Solver": solver_name,
            "Sample Steps": sample_steps,
            "Shift": shift,
            "Video Guidance Scale": video_cfg_scale,
            "Video Ref Guidance Scale": video_ref_cfg_scale,
            "Audio Guidance Scale": audio_cfg_scale,
            "Audio Ref Guidance Scale": audio_ref_cfg_scale,
            "Video Negative Prompt": video_negative_prompt,
            "Audio Negative Prompt": audio_negative_prompt,
        }
        pretty = "\n".join(f"{k:>24}: {v}" for k, v in params.items())
        logging.info("\n========== Generation Parameters ==========\n"
                    f"{pretty}\n"
                    "==========================================")

        try:

            # paths = [image0_path, image1_path, audio0_path, audio1_path]
            raw_paths = [image0_path, image1_path, audio0_path, audio1_path]
            paths = [p for p in raw_paths if p is not None]
            ref_vae_latents, ref_audio_lengths = self.load_image_latent_ref_ip_video(
                paths, video_frame_height_width, self.device
            )
            latents_ref_image = ref_vae_latents["image"]
            latents_ref_audio = ref_vae_latents["audio"]
            ref_ip_num = latents_ref_image.shape[1]
            ref_audio_length = latents_ref_audio.shape[0]
       
     
            scheduler_video, timesteps_video = self.get_scheduler_time_steps(
                sampling_steps=sample_steps, device=self.device, solver_name=solver_name, shift=shift
            )
            scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
                sampling_steps=sample_steps, device=self.device, solver_name=solver_name, shift=shift
            )
  
            if self.cpu_offload: 
                self.text_model.model.to(self.device)
            text_embeddings = self.text_model([text_prompt, video_negative_prompt, audio_negative_prompt], self.text_model.device)
            text_embeddings = [emb.to(self.target_dtype).to(self.device) for emb in text_embeddings]
            if self.cpu_offload: 
                self.offload_to_cpu(self.text_model.model)
            text_embeddings_audio_pos = text_embeddings[0]
            text_embeddings_video_pos = text_embeddings[0]
            text_embeddings_video_neg = text_embeddings[1]
            text_embeddings_audio_neg = text_embeddings[2]
     
            video_h, video_w = video_frame_height_width
            video_latent_h, video_latent_w = video_h // 16, video_w // 16

            video_noise_len = self.video_latent_length + ref_ip_num
            audio_noise_len = self.audio_latent_length + ref_audio_length
            freqs_scaling_tensor = torch.tensor(self.video_latent_length / self.audio_latent_length, device=self.device, dtype=self.target_dtype)

            video_noise = torch.randn((self.video_latent_channel, video_noise_len, video_latent_h, video_latent_w), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))
            audio_noise = torch.randn((audio_noise_len, self.audio_latent_channel), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))
 
            _patch_size_h, _patch_size_w = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]
         
            max_seq_len_video = video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (_patch_size_h*_patch_size_w)
    
            max_seq_len_audio = audio_noise_len


          
            if self.cpu_offload: # Offload VAEs and put DiT on device
                self.offload_to_cpu(self.vae_model_video.model)
                self.offload_to_cpu(self.vae_model_audio)
                self.model = self.model.to(self.device)
       
            with torch.amp.autocast('cuda', enabled=self.target_dtype != torch.float32, dtype=self.target_dtype):
                for _, (t_v, t_a) in tqdm(enumerate(zip(timesteps_video, timesteps_audio))):
                    timestep_input = torch.full((1,), t_v, device=self.device)

                    model_input_video = torch.cat([video_noise[:, :-ref_ip_num], latents_ref_image], dim=1)
                    model_input_video_neg = torch.cat([video_noise[:, :-ref_ip_num], torch.zeros_like(latents_ref_image)], dim=1)
 
                    model_input_audio = torch.cat([audio_noise[:-ref_audio_length, :], latents_ref_audio], dim=0)
            
                    model_input_audio_neg = torch.cat([audio_noise[:-ref_audio_length, :], torch.zeros_like(latents_ref_audio)], dim=0)
                    ref_ip_lengths = [ref_ip_num]
        
                    common_args = {
                        'vid_seq_len': max_seq_len_video,
                        'audio_seq_len': max_seq_len_audio,
                        'freqs_scaling': freqs_scaling_tensor,
                        'ref_ip_lengths': [ref_ip_lengths],
                        'ref_audio_lengths': [ref_audio_lengths],
                    }
                    pos_args = {**common_args, 'audio_context': [text_embeddings_audio_pos], 'vid_context': [text_embeddings_video_pos]}
                    neg_args = {**common_args, 'audio_context': [text_embeddings_audio_neg], 'vid_context': [text_embeddings_video_neg]}

                    pred_vid_pos, pred_audio_pos = self.model(vid=[model_input_video], audio=[model_input_audio], t=timestep_input, **pos_args)

                    pred_vid_neg, pred_audio_neg = self.model(vid=[model_input_video], audio=[model_input_audio], t=timestep_input, **neg_args)

                    pre_vid_ip_neg, _ = self.model(vid=[model_input_video_neg], audio=[model_input_audio], t=timestep_input, **pos_args)

                    _, pred_refaudio_neg = self.model(vid=[model_input_video], audio=[model_input_audio_neg], t=timestep_input, **pos_args)
     
                    pred_video_guided = pred_vid_neg[0] + \
                                        video_cfg_scale * (pred_vid_pos[0] - pred_vid_neg[0]) + \
                                        video_ref_cfg_scale * (pred_vid_pos[0] - pre_vid_ip_neg[0])

                    pred_audio_guided = pred_audio_neg[0] + \
                                        audio_cfg_scale * (pred_audio_pos[0] - pred_audio_neg[0]) + \
                                        audio_ref_cfg_scale * (pred_audio_pos[0] - pred_refaudio_neg[0])
                    video_noise = scheduler_video.step(pred_video_guided.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False)[0].squeeze(0)
                    audio_noise = scheduler_audio.step(pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False)[0].squeeze(0)
            
            torch.cuda.empty_cache()
            if self.cpu_offload: # Offload DiT and put VAEs on device
                self.offload_to_cpu(self.model)
                self.vae_model_video.model = self.vae_model_video.model.to(self.device)
                self.vae_model_audio = self.vae_model_audio.to(self.device)
   
            video_noise_for_decode = video_noise[:, :-ref_ip_num]
            audio_noise_for_decode = audio_noise[:-ref_audio_length, :]

            audio_latents_for_vae = audio_noise_for_decode.unsqueeze(0).transpose(1, 2)
            generated_audio = self.vae_model_audio.wrapped_decode(audio_latents_for_vae).squeeze().cpu().float().numpy()

            del audio_latents_for_vae

            video_latents_for_vae = video_noise_for_decode.unsqueeze(0)
            generated_video = self.vae_model_video.wrapped_decode(video_latents_for_vae).squeeze(0).cpu().float().numpy()
            if self.cpu_offload:
                self.offload_to_cpu(self.vae_model_video.model)
            del video_latents_for_vae
            
            return generated_video, generated_audio, None



        except Exception as e:
            logging.error(traceback.format_exc())
            return None
            
    def offload_to_cpu(self, model):
        model = model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return model

    def get_scheduler_time_steps(self, sampling_steps, solver_name='unipc', device=0, shift=5.0):
        torch.manual_seed(4)

        if solver_name == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps

        elif solver_name == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)
            
        elif solver_name == 'euler':
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                shift=shift
            )
            timesteps, sampling_steps = retrieve_timesteps(
                sample_scheduler,
                sampling_steps,
                device=device,
            )
        
        else:
            raise NotImplementedError("Unsupported solver.")
        
        return sample_scheduler, timesteps