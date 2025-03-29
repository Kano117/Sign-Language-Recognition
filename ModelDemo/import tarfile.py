import torch
import numpy as np
import cv2
from model import SignLanguageModel 

cfg = {
    "task": "S2G",
    "device": "cpu",
    "do_translation": False,
    "do_recognition": True,
    "data": {
        "train_label_path": "data/CSL-Daily/CSL-Daily.train",
        "dev_label_path": "data/CSL-Daily/CSL-Daily.dev",
        "test_label_path": "data/CSL-Daily/CSL-Daily.test",
        "max_length": 400,
        "dataset_name": "CSL-Daily",
        "input_streams": "keypoint",
        "level": "word",
        "txt_lowercase": True,
        "max_sent_length": 400,
    },
    "gloss": {
        "gloss2id_file": "data/CSL-Daily/gloss2ids.pkl",
    },
    "testing": {
        "recognition": {
            "beam_size": 5,
        },
        "translation": {},
    },
    "training": {
        "wandb": "disabled",
        "model_dir": "outputs/CSL-Daily_SLR",
        "validation": {
            "recognition": {
                "beam_size": 1,
            },
            "translation": {},
        },
        "optimization": {
            "optimizer": "Adam",
            "learning_rate": {
                "default": 1.0e-3,
            },
            "weight_decay": 0.001,
            "betas": [0.9, 0.998],
            "scheduler": "cosineannealing",
            "t_max": 100,
        },
    },
    "model": {
        "RecognitionNetwork": {
            "input_type": "keypoint",
            "DSTA-Net": {
                "net": [
                    [64, 64, 16, 7, 2],
                    [64, 64, 16, 3, 1],
                    [64, 128, 32, 3, 1],
                    [128, 128, 32, 3, 1],
                    [128, 256, 64, 3, 2],
                    [256, 256, 64, 3, 1],
                    [256, 256, 64, 3, 1],
                    [256, 256, 64, 3, 1],
                ],
                "body": [0, 1, 3, 5, 7, 9, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 2, 4, 6, 8, 10, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 23, 26, 29, 33, 36, 39, 41, 43, 46, 48, 53, 56, 59, 62, 65, 68, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81],
                "left": [0, 1, 3, 5, 7, 9, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "right": [0, 2, 4, 6, 8, 10, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132],
                "face": [23, 26, 29, 33, 36, 39, 41, 43, 46, 48, 53, 56, 59, 62, 65, 68, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81],
                "mouth": [71, 72, 73, 74, 75, 76, 77, 79, 80, 81],
            },
            "GlossTokenizer": {
                "gloss2id_file": "data/CSL-Daily/gloss2ids.pkl",
            },
            "body_visual_head": {
                "input_size": 256,
                "hidden_size": 512,
                "ff_size": 2048,
                "pe": True,
                "ff_kernelsize": [3, 3],
            },
            "fuse_visual_head": {
                "input_size": 1024,
                "hidden_size": 512,
                "ff_size": 2048,
                "pe": True,
                "ff_kernelsize": [3, 3],
            },
            "left_visual_head": {
                "input_size": 512,
                "hidden_size": 512,
                "ff_size": 2048,
                "pe": True,
                "ff_kernelsize": [3, 3],
            },
            "right_visual_head": {
                "input_size": 512,
                "hidden_size": 512,
                "ff_size": 2048,
                "pe": True,
                "ff_kernelsize": [3, 3],
            },
            "cross_distillation": True,
        },
        "VLMapper": {
            "type": "projection",
            "in_features": 512,
        },
        "TranslationNetwork": {
            "input_dim": 512,
        },
    },
}

args = {
    "learning_rate": 0.001,
    "batch_size": 32,
}

model = SignLanguageModel(cfg, args)
pre_trained_path = "D:\\Tailieu\\HK5\\DoAn\\Modelkhaosat\\Model\\SLR\\best.pth"
checkpoint = torch.load(pre_trained_path, map_location=torch.device('cpu'))
state_dict = checkpoint['model']
model.load_state_dict(state_dict, strict=False)
print(f'Model {pre_trained_path} loaded')

cap = cv2.VideoCapture(0)

with torch.no_grad():
    model.eval()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_frame = cv2.resize(frame, (224, 224))
        input_frame = input_frame / 255.0
        input_frame = np.transpose(input_frame, (2, 0, 1))
        input_frame = np.expand_dims(input_frame, axis=0)
        input_tensor = torch.tensor(input_frame, dtype=torch.float32)
        print(f"Input tensor shape before permute: {input_tensor.shape}")
        mask_tensor = torch.ones((1, 400))  
        length_tensor = torch.tensor([56])  
        gloss_labels = torch.tensor([[1, 2, 3]])  
        gloss_lengths = torch.tensor([3])  

        print("Mask shape:", mask_tensor.shape)
        print("Length tensor shape:", length_tensor.shape)
        print("Gloss labels shape:", gloss_labels.shape)
        print("Gloss lengths shape:", gloss_lengths.shape)

        src_input = {
            'keypoint': input_tensor,  
            'mask': mask_tensor,
            'new_src_lengths': length_tensor,
            'gloss_input': {'gloss_labels': gloss_labels, 'gls_lengths': gloss_lengths}
                    }

        output_tensor = src_input['keypoint']  
        class_index = torch.argmax(output_tensor).item()
        cv2.putText(frame, f'Class: {class_index}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()