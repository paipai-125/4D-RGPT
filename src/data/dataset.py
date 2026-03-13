import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import random
import numpy as np
import torchvision
from PIL import Image

class RGPTDataset(Dataset):
    def __init__(self, data_root, mode='train', max_frames=16):
        self.data_root = data_root
        self.mode = mode
        self.max_frames = max_frames
        self.data = []
        
        # Load from JSON if exists, else verify_pipeline mocks
        json_path = os.path.join(data_root, 'training_qa.json') if mode == 'train' else os.path.join(data_root, 'test_qa_sim/test_qa.json')
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self.data = json.load(f)
        else:
            # For verifying pipleine without full data
            print(f"Warning: JSON not found at {json_path}. Using dummy data.")
            self.data = [{"dummy": True}] * 10

    def __len__(self):
        return len(self.data)

    def _load_video(self, video_path):
        target_res = 384
        if not os.path.exists(video_path):
            # print(f"Video missing: {video_path}")
            T, C, H, W = self.max_frames, 3, target_res, target_res
            return torch.zeros(T, C, H, W)
            
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames > 0:
                indices = np.linspace(0, total_frames - 1, self.max_frames).astype(int)
                frames = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (target_res, target_res))
                        frames.append(frame)
                    else:
                        pass
                
                cap.release()
                
                if len(frames) > 0:
                    # Stack: [T, H, W, C]
                    video = np.stack(frames)
                    # To [T, C, H, W]
                    video = torch.from_numpy(video).permute(0, 3, 1, 2)
                    video = video.float() / 255.0
                    
                    if video.shape[0] < self.max_frames:
                         padding = torch.zeros(self.max_frames - video.shape[0], 3, target_res, target_res)
                         video = torch.cat([video, padding], dim=0)
                         
                    return video
            
            cap.release()
            return torch.zeros(self.max_frames, 3, target_res, target_res)

        except ImportError:
            # Fallback to torchvision if cv2 is missing
            try:
                video, _, _ = torchvision.io.read_video(video_path, pts_unit='sec', output_format="TCHW")
                total_frames = video.shape[0]
                if total_frames > 0:
                    if total_frames > self.max_frames:
                        indices = torch.linspace(0, total_frames - 1, self.max_frames).long()
                        video = video[indices]
                    
                    video = video.float() / 255.0
                    video = torch.nn.functional.interpolate(video, size=(target_res, target_res), mode='bilinear', align_corners=False)
                    return video
                else:
                    return torch.zeros(self.max_frames, 3, target_res, target_res)
            except Exception as e:
                return torch.zeros(self.max_frames, 3, target_res, target_res)
                
        except Exception as e:
            return torch.zeros(self.max_frames, 3, target_res, target_res)

    def __getitem__(self, idx):
        item = self.data[idx]
        target_res = 384
        
        # Mock Data Generation
        if "dummy" in item:
            # Create random video tensor: [T, C, H, W]
            T, C, H, W = self.max_frames, 3, target_res, target_res
            video = torch.randn(T, C, H, W)
            
            # Create Mock Timestamp
            timestamp = float(random.randint(0, 10))
            
            # Create Mock Text Prompts
            text = "Move the cube to the right."
            
            return {
                "video": video,
                "timestamp": torch.tensor(timestamp, dtype=torch.float),
                "text": text,
                "dataset_name": "dummy"
            }

        # Real Data Loading Attempt
        # Assuming Data format has 'video_path', 'instruction', 'timestamp'
        video_rel_path = item.get('video_path', '') # Adjust key based on json structure
        video_full_path = os.path.join(self.data_root, video_rel_path)
        
        video = self._load_video(video_full_path)
        
        # Parse timestamp if needed, or get from json
        # Assuming json has 'time' float field
        timestamp = item.get('time', 0.0)
        
        text = item.get('instruction', '') # Adjust key
        
        return {
            "video": video,
            "timestamp": torch.tensor(timestamp, dtype=torch.float),
            "text": text,
            "dataset_name": "RoboFAC"
        }

def collate_fn(batch):
    videos = torch.stack([item['video'] for item in batch])
    timestamps = torch.stack([item['timestamp'] for item in batch])
    texts = [item['text'] for item in batch]
    dataset_names = [item['dataset_name'] for item in batch]
    
    return {
        "pixel_values": videos, # Compatible name for models
        "timestamps": timestamps,
        "input_ids": texts, # Needs tokenization in Training loop or here
        "dataset_names": dataset_names
    }
