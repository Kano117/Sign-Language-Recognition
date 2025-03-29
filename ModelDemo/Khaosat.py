import torch
import numpy as np
from model import SignLanguageModel 

cfg = {
    "task": "S2G",  # or "S2T" depending on your task
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model": {
        "RecognitionNetwork": {
            "input_type": "keypoint",  # Specify the input type
            "GlossTokenizer": {
                "gloss2id_file": "D:\Tailieu\HK5\DoAn\Modelkhaosat\Khaosattongthe\gloss2ids.pkl",  # Path to the gloss2id file
                "lower_case": True  # Optional, default is True
            },
            "DSTA-Net": {
                "net": [
                    (3, 64, 32, 3, 1),  # (in_channels, out_channels, inter_channels, t_kernel, stride)
                    (64, 128, 64, 3, 2),
                    (128, 256, 128, 3, 2),
                ],
                "left": [0, 1, 2],  # Example indices for left keypoints
                "right": [3, 4, 5],  # Example indices for right keypoints
                "face": [6, 7, 8],  # Example indices for face keypoints
                "body": [9, 10, 11],  # Example indices for body keypoints
            },
            "fuse_visual_head": {
                "hidden_size": 512,  # Ensure this is an even number
                "num_classes": 100,  # Replace with the actual number of classes
            },
            "body_visual_head": {
                "hidden_size": 512,  # Ensure this is an even number
                "num_classes": 100,  # Replace with the actual number of classes
            },
            "left_visual_head": {
                "hidden_size": 512,  # Ensure this is an even number
                "num_classes": 100,  # Replace with the actual number of classes
            },
            "right_visual_head": {
                "hidden_size": 512,  # Ensure this is an even number
                "num_classes": 100,  # Replace with the actual number of classes
            },
        },
        "VLMapper": {
            "type": "projection",  # Type of VLMapper
            "in_features": 512,  # Input feature size
        },
        "TranslationNetwork": {
            "input_dim": 512,  # Input dimension for the translation network
        },
    },
}

args = {
    "learning_rate": 0.001,  # Learning rate for training
    "batch_size": 32,        # Batch size for training
}


print("Args:", args)
model = SignLanguageModel(cfg, args)
model.load_state_dict(torch.load('D:\\Tailieu\\HK5\\DoAn\\Modelkhaosat\\MSKA-main\\pretrained_models\\best.pth'))

with torch.no_grad():
    model.eval()
    result = model('D:\\Tailieu\\HK5\\DoAn\\Modelkhaosattongthe\\01April_2010_Thursday_tagesschau-4332\\images0001.png')