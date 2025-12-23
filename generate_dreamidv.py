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

import argparse
from datetime import datetime
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image, ImageOps

import dreamidv_wan
from dreamidv_wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from dreamidv_wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from dreamidv_wan.utils.utils import cache_video, cache_image, str2bool

import cv2
import numpy as np
from express_adaption.media_pipe import FaceMeshDetector, FaceMeshAlign_dreamidv
from express_adaption.get_video_npy import get_video_npy


def generate_pose_and_mask_videos(ref_video_path, ref_image_path):

    print("Starting online generation of pose and mask videos...")
    detector = FaceMeshDetector()
    get_align_motion = FaceMeshAlign_dreamidv()
    CORE_LANDMARK_INDICES = [
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        1, 2, 5, 6, 48, 64, 94, 98, 168, 195, 197, 278, 294, 324, 327, 4, 24,
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
        468, 473, 55, 65, 52, 53, 46, 285, 295, 282, 283, 276, 70, 63, 105, 66, 107,
        300, 293, 334, 296, 336, 156,
    ]
    FACE_OVAL_INDICES = [
        10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    ]
    CORE_LANDMARK_INDICES.extend(FACE_OVAL_INDICES)
    CORE_LANDMARK_INDICES = list(set(CORE_LANDMARK_INDICES))
    def save_visualization_video(landmarks_sequence, output_filename, frame_size, fps=30, mode='points'):
        width, height = frame_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {output_filename}")
            return
        print(f"Saving {mode} video to {output_filename}...")
        for frame_landmarks in landmarks_sequence:
            frame_image = np.zeros((height, width, 3), dtype=np.uint8)
            if mode == 'points':
                for landmark in frame_landmarks:
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(frame_image, (x, y), radius=2, color=(255, 255, 255), thickness=-1)
            elif mode == 'mask':
                face_oval_points = frame_landmarks.astype(np.int32)
                cv2.fillConvexPoly(frame_image, face_oval_points, color=(255, 255, 255))
            video_writer.write(frame_image)
        video_writer.release()
        print("Video saving complete.")
    fps = cv2.VideoCapture(ref_video_path).get(cv2.CAP_PROP_FPS)
    face_results = get_video_npy(ref_video_path)
    video_name = os.path.basename(ref_video_path).split('.')[0]
    temp_dir = os.path.join(os.path.dirname(ref_video_path), 'temp_generated')
    os.makedirs(temp_dir, exist_ok=True)
    image = Image.open(ref_image_path).convert('RGB')
    ref_image = np.array(image)
    _, ref_img_lmk = detector(ref_image)
    _, pose_addvis = get_align_motion(face_results, ref_img_lmk)
    width, height = face_results[0]['width'], face_results[0]['height']
 
    pose_output_path = os.path.join(temp_dir, video_name + '_pose.mp4')
    core_landmarks_sequence = pose_addvis[:, CORE_LANDMARK_INDICES, :]
    save_visualization_video(
        landmarks_sequence=core_landmarks_sequence,
        output_filename=pose_output_path,
        frame_size=(width, height),
        fps=fps,
        mode='points'
    )
    mask_output_path = os.path.join(temp_dir, video_name + '_mask.mp4')
    face_oval_sequence = pose_addvis[:, FACE_OVAL_INDICES, :]
    save_visualization_video(
        landmarks_sequence=face_oval_sequence,
        output_filename=mask_output_path,
        frame_size=(width, height),
        fps=fps,
        mode='mask'
    )
    return pose_output_path, mask_output_path



def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.dreamidv_ckpt is not None, "Please specify the Phantom-Wan checkpoint."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 12

    if args.sample_shift is None:
        args.sample_shift = 5.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 81


    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="swapface",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--sample_fps",
        type=int,
        default=24,
        help="The fps of the generated video."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--dreamidv_ckpt",
        type=str,
        default=None,
        help="The path to the Phantom-Wan checkpoint.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="ch",
        choices=["ch", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--ref_image",
        type=str,
        default='./assets/test_case/ref_image/an_1.jpg',
        help="The reference images used by DreamID-V.")
    parser.add_argument(
        "--ref_video",
        type=str,
        default='./assets/test_case/ref_video/a_girl.mp4',
        help="The reference video used by DreamID-V.")

    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale_img",
        type=float,
        default=4.0,
        help="Classifier free guidance scale for reference images.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)





def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    if args.sample_fps is not None:
        cfg.sample_fps = args.sample_fps

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

 
    logging.info("Creating DreamID-V pipeline.")
    wan_swapface = dreamidv_wan.DreamIDV(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        dreamidv_ckpt=args.dreamidv_ckpt,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )

    
    prompt = 'chang face'
    try:
        ref_pose_path, ref_mask_path = generate_pose_and_mask_videos(
            ref_video_path=args.ref_video,
            ref_image_path=args.ref_image
        )
    except:
        print("Pose and mask video generation failed. no pose detected in the reference video.")
    ref_video_path = args.ref_video
    ref_img_path = args.ref_image
    text_prompt = prompt
        
    ref_paths = [
        ref_video_path,
        ref_mask_path,
        ref_img_path,
        ref_pose_path
    ]

    video = wan_swapface.generate(
        text_prompt,
        ref_paths,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale_img=args.sample_guide_scale_img,
        seed=args.base_seed,
        offload_model=args.offload_model)


    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_time}" + suffix
        cache_video(
                    tensor=video[None],
                    save_file=args.save_file,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
        print(f"Save file: {args.save_file}")

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
