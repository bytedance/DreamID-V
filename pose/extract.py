# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import cv2
import torch
import numpy as np
import imageio
from tqdm import tqdm
from PIL import Image



def convert_to_numpy(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        image = image.copy()
    else:
        raise TypeError(f'Unsupported datatype {type(image)}')
    return image

def read_video_frames(video_path, use_type='cv2', is_rgb=True, info=False):
    frames = []
    if use_type == "cv2":
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"Reading video: {video_path}")
            print(f"Original Info -> Size: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
            
            pbar = tqdm(total=total_frames, desc="Reading Video")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if is_rgb:
                    frames.append(frame[..., ::-1])
                else:
                    frames.append(frame)
                pbar.update(1)
            pbar.close()
            cap.release()
        except Exception as e:
            print(f"OpenCV read error: {e}")
            return None
    else:
        raise ValueError(f"Unknown type {use_type}")

    if info:
        return frames, fps, width, height, len(frames)
    else:
        return frames

def save_one_video(file_path, videos, fps=30, quality=8):
    print(f"Saving video to {file_path}...")
    try:
        
        video_writer = imageio.get_writer(
            file_path, 
            fps=fps, 
            codec='libx264', 
            quality=quality,
            macro_block_size=None 
        )
        for frame in tqdm(videos, desc="Saving Video"):
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if len(frame.shape) == 2:
                frame = np.stack([frame]*3, axis=-1)
            video_writer.append_data(frame)
        video_writer.close()
        return True
    except Exception as e:
        print(f"Video save error: {e}")
        return False



try:
    from dwpose import util as dwpose_util
    from dwpose.wholebody import Wholebody, HWC3, resize_image
except ImportError:
    raise ImportError("no dwpose module found")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def draw_pose(pose, H, W, use_hand=False, use_body=False, use_face=False):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if use_body:
        canvas = dwpose_util.draw_bodypose(canvas, candidate, subset)
    if use_hand:
        canvas = dwpose_util.draw_handpose(canvas, hands)
    if use_face:
        canvas = dwpose_util.draw_facepose(canvas, faces)

    return canvas

def draw_face_mask_from_points(faces, H, W):
    mask = np.zeros((H, W, 3), dtype=np.uint8)
  
    max_area = 0
    best_hull = None
    for face in faces:
        valid_points = []
        for pt in face:
            if pt[0] >= 0 and pt[1] >= 0:
                x = int(pt[0] * W)
                y = int(pt[1] * H)
                valid_points.append([x, y])
        if len(valid_points) > 3:
            pts = np.array(valid_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            hull = cv2.convexHull(pts)
     
            current_area = cv2.contourArea(hull)
            if current_area > max_area:
                max_area = current_area
                best_hull = hull

    if best_hull is not None:
        cv2.fillPoly(mask, [best_hull], (255, 255, 255))

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

class PoseAnnotator:
    def __init__(self, cfg, device=None):
        onnx_det = cfg['DETECTION_MODEL']
        onnx_pose = cfg['POSE_MODEL']

        if device is None:
            if hasattr(torch, 'npu') and torch.npu.is_available():
                self.device = torch.device("npu")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")

        self.pose_estimation = Wholebody(onnx_det, onnx_pose, device=self.device)
        self.resize_size = cfg.get("RESIZE_SIZE", 1024)
        self.use_body = cfg.get('USE_BODY', True)
        self.use_face = cfg.get('USE_FACE', True)
        self.use_hand = cfg.get('USE_HAND', True)

    @torch.no_grad()
    @torch.inference_mode()
    def forward(self, image):
        image = convert_to_numpy(image)
        input_image = HWC3(image)
        return self.process(resize_image(input_image, self.resize_size), image.shape[:2])

    def process(self, input_data, ori_shape):
        ori_img = input_data
        ori_h, ori_w = ori_shape 
        H, W, C = ori_img.shape  

        with torch.no_grad():
            candidate, subset, det_result = self.pose_estimation(ori_img)
            nums, keys, locs = candidate.shape
            

            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            score = subset[:, :18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1

            faces = candidate[:, 24:92]
            
            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            ret_data = {}

            def get_resized_pose_map(use_body=False, use_face=False, use_hand=False):
                temp_map = draw_pose(pose, H, W, use_body=use_body, use_face=use_face, use_hand=use_hand)
      
                resized_map = cv2.resize(temp_map, (ori_w, ori_h), interpolation=cv2.INTER_LANCZOS4)
                return resized_map

            if self.use_body:
                ret_data["detected_map_body"] = get_resized_pose_map(use_body=True)

            if self.use_body and self.use_face:
                ret_data["detected_map_bodyface"] = get_resized_pose_map(use_body=True, use_face=True)
            face_mask = draw_face_mask_from_points(faces, ori_h, ori_w)
            ret_data["face_mask"] = face_mask

            return ret_data, det_result

class PoseBodyFaceVideoAnnotator(PoseAnnotator):
    def __init__(self, cfg, device=None):
        super().__init__(cfg, device)
        self.use_body = cfg.get('USE_BODY', True)
        self.use_face = cfg.get('USE_FACE', True)
        self.use_hand = cfg.get('USE_HAND', False)

    def forward_video(self, frames):
        pose_frames = []
        mask_frames = []
        
        print("Processing frames for Pose & Face Mask...")
        for frame in tqdm(frames, desc="Inference"):
            ret_data, _ = self.forward(np.array(frame))

     
            if "detected_map_bodyface" in ret_data:
                pose_frame = ret_data['detected_map_bodyface']
            elif "detected_map_body" in ret_data:
                pose_frame = ret_data['detected_map_body']
            else:
                h, w = frame.shape[:2]
                pose_frame = np.zeros((h, w, 3), dtype=np.uint8)
            
      
            if "face_mask" in ret_data:
                mask_frame = ret_data['face_mask']
            else:
                h, w = frame.shape[:2]
                mask_frame = np.zeros((h, w, 3), dtype=np.uint8)

            pose_frames.append(pose_frame)
            mask_frames.append(mask_frame)
            
        return pose_frames, mask_frames



def process_dwpose(input_video_path, output_pose_path, output_mask_path):

    current_dir = os.path.dirname(os.path.abspath(__file__))

    det_model_path = os.path.join(current_dir, 'models/yolox_l.onnx')
    pose_model_path = os.path.join(current_dir, 'models/dw-ll_ucoco_384.onnx')
    config = {
        'DETECTION_MODEL': det_model_path,
        'POSE_MODEL': pose_model_path,
        'RESIZE_SIZE': 1024,
        'USE_BODY': True,
        'USE_FACE': True,
        'USE_HAND': False
    }
    if not os.path.exists(input_video_path):
        print(f"Error: Input video not found at {input_video_path}")
        return
    if not os.path.exists(config['DETECTION_MODEL']) or not os.path.exists(config['POSE_MODEL']):
        print(f"Error: Model files not found in {current_dir}/models/")
        return

    frames, fps, width, height, total = read_video_frames(input_video_path, use_type='cv2', info=True)
    if not frames:
        print("Failed to read video.")
        return

    try:
        annotator = PoseBodyFaceVideoAnnotator(config)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    pose_frames, mask_frames = annotator.forward_video(frames)

    print(f"Saving Pose Video to {output_pose_path}...")
    save_one_video(output_pose_path, pose_frames, fps=fps)
    print(f"Saving Face Mask Video to {output_mask_path}...")
    save_one_video(output_mask_path, mask_frames, fps=fps)
    print("DWPose Generation Done!")
if __name__ == "__main__":

    test_video = "test.mp4"
    if os.path.exists(test_video):
        process_dwpose(test_video, "out_pose.mp4", "out_mask.mp4")