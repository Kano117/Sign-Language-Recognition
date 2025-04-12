import cv2
import pickle
import torch
import mmcv
import numpy as np
from mmpose.apis import inference_topdown, init_model

# Load model HRNet 
config_file = 'td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py'
checkpoint_file = 'hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = init_model(config_file, checkpoint_file, device=device)

# Load video 
video_path = '4329.mp4'
video = mmcv.VideoReader(video_path)

keypoints = []
for frame in video:
    results = inference_topdown(model, frame)

    if results and results[0].pred_instances.keypoints.shape[0] > 0:
        kp_xy = results[0].pred_instances.keypoints[0]         
        scores = results[0].pred_instances.keypoint_scores[0]  
        kp_combined = np.concatenate([kp_xy, scores[..., None]], axis=-1)  
        keypoints.append(torch.tensor(kp_combined, dtype=torch.float32))
    else:
        keypoints.append(torch.zeros((133, 3), dtype=torch.float32))  

# Tạo sample
sample_dict = {
    'keypoint': torch.stack(keypoints),  
    'gloss': 'NULL',
    'text': None,
    'num_frames': len(keypoints),
    'name': '4329'
}

data_dict = {
    'test/4329': sample_dict
}

with open('4329.test', 'wb') as f:
    pickle.dump(data_dict, f)

print(" Đã tạo file .test ")
