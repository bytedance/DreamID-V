# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from poseguider import Guider
import torch
from PIL import Image
from media_pipe import FaceMeshDetector, FaceMeshAlign_dreamidv
import numpy as np
from diffusers.image_processor import VaeImageProcessor
import cv2
import os
from get_video_npy import get_video_npy


CORE_LANDMARK_INDICES = [
    # 嘴巴 (Mouth) - 40个点
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    95, 88, 178, 87, 14, 317, 402, 318, 324,
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    # 鼻子 (Nose) - 19个点
    1, 2, 5, 6, 48, 64, 94, 98, 168, 195, 197, 278, 294, 324, 327,
    4, 24,
    # 眼睛 (Eyes) - 32个点 (左右各16个)
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, # 右眼
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, # 左眼
    # 虹膜 (Irises) - 2个点 (左右各1个)
    468, # 右虹膜
    473, # 左虹膜
    # 眉毛 (Eyebrows) - 25个点
    55, 65, 52, 53, 46,
    285, 295, 282, 283, 276,
    70, 63, 105, 66, 107,
    300, 293, 334, 296, 336,
    156,
]

FACE_OVAL_INDICES = [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
]

CORE_LANDMARK_INDICES.extend(FACE_OVAL_INDICES)

CORE_LANDMARK_INDICES = list(set(CORE_LANDMARK_INDICES))

face_oval_map = [i for i, _ in enumerate(FACE_OVAL_INDICES)]



def save_visualization_video(landmarks_sequence, output_filename, frame_size, fps=30, mode='points', optional_face_oval_indices=None):
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
        elif mode == 'mask' and optional_face_oval_indices is not None:
            face_oval_points = frame_landmarks[optional_face_oval_indices].astype(np.int32)
            cv2.fillConvexPoly(frame_image, face_oval_points, color=(255, 255, 255))
        video_writer.write(frame_image)
    video_writer.release()
    print("Video saving complete.")

detector = FaceMeshDetector()
get_align_motion = FaceMeshAlign_dreamidv()
cond_image_processor = VaeImageProcessor(vae_scale_factor=1, do_convert_rgb=True, do_normalize=False)
weight_path = 'express_adaption/weight/lmk_guider.pth'
lmk_guider = Guider(conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256))
lmk_guider.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
lmk_guider.eval()


img_path = 'crop_imgs/male_ref1.jpeg'
video_path = 'videos/male_006.mp4'

fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
face_results = get_video_npy(video_path)
video_name = os.path.basename(video_path).split('.')[0]
align_pose_root_path = 'video_face_swap_test/align_pose'
os.makedirs(align_pose_root_path, exist_ok=True)
image = Image.open(img_path).convert('RGB')

ref_image = np.array(image)
_, ref_img_lmk = detector(ref_image)

driving_video_lmks = face_results


_, pose_addvis = get_align_motion(driving_video_lmks, ref_img_lmk) 



width, height = driving_video_lmks[0]['width'], driving_video_lmks[0]['height']

core_landmarks_sequence = pose_addvis[:, CORE_LANDMARK_INDICES, :]

save_visualization_video(
    landmarks_sequence=core_landmarks_sequence,
    output_filename=os.path.join(align_pose_root_path, video_name + '_pose.mp4'), 
    frame_size=(width, height),
    fps=fps,
    mode='points' 
)

face_oval_sequence = pose_addvis[:, FACE_OVAL_INDICES, :]

save_visualization_video(
    landmarks_sequence=face_oval_sequence, 
    output_filename=os.path.join(align_pose_root_path, video_name + '_mask.mp4'), 
    frame_size=(width, height),
    fps=fps,
    mode='mask', 
    optional_face_oval_indices=np.arange(len(FACE_OVAL_INDICES)) 
)






