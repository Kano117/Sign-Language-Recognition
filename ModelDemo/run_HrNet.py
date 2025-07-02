import cv2
import pickle
import torch
import mmcv
import yaml
import time
import numpy as np
from pathlib import Path
from mmpose.apis import inference_topdown, init_model
from model import SignLanguageModel
from Tokenizer import GlossTokenizer_S2G
from datasets import S2T_Dataset

def extract_keypoints_from_video(video_path, output_test_path):
    # Load HRNet model
    config_file = 'td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py'
    checkpoint_file = 'hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = init_model(config_file, checkpoint_file, device=device)

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

    name = Path(video_path).stem
    sample_dict = {
        'keypoint': torch.stack(keypoints),
        'gloss': 'NULL',
        'text': None,
        'num_frames': len(keypoints),
        'name': name
    }

    data_dict = {f'test/{name}': sample_dict}
    with open(output_test_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Đã tạo file .test tại: {output_test_path}")
    return name

def update_config_test_path(config_path, new_test_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['data']['test_label_path'] = new_test_path
    return config

def run_model_on_test(config, test_index=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GlossTokenizer_S2G(config['gloss'])

    test_data = S2T_Dataset(
        path=config['data']['test_label_path'],
        tokenizer=tokenizer,
        config=config,
        args=None,
        phase='test',
        training_refurbish=False
    )

    sample = test_data[test_index]
    sample_input = test_data.collate_fn([sample])

    model = SignLanguageModel(cfg=config, args=None)
    checkpoint_path = 'pretrained_models\\best.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for key in sample_input:
            if isinstance(sample_input[key], torch.Tensor):
                sample_input[key] = sample_input[key].to(device)

        output = model(sample_input)
        gloss_logits = output['ensemble_last_gloss_logits']
        input_lengths = output['input_lengths']
        decoded_ids = model.recognition_network.decode(gloss_logits, beam_size=5, input_lengths=input_lengths)
        gloss_prediction = tokenizer.convert_ids_to_tokens(decoded_ids)[0]

        print("\n===== KẾT QUẢ NHẬN DIỆN =====")
        print("Predicted Glosses:", ' '.join(gloss_prediction))
        print("Reference Glosses:", sample_input.get('gloss', ['NULL'])[0])

# === Loop Webcam 4s Ghi - Nhận Diện ===
if __name__ == "__main__":
    config_path = 'configs/phoenix-2014t_s2g.yaml'
    test_file = 'temp_webcam.test'
    cam_index = 0
    VIDEO_DURATION = 4  # giây
    FPS = 15
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    print("Nhấn Q để thoát.")

    while True:
        out_path = 'webcam_temp.mp4'
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (FRAME_WIDTH, FRAME_HEIGHT))

        print("Ghi video 4 giây...")
        start_time = time.time()
        while time.time() - start_time < VIDEO_DURATION:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            cv2.imshow("Webcam Recording...", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # ➤ Nhận diện
        name = extract_keypoints_from_video(out_path, test_file)
        config = update_config_test_path(config_path, test_file)
        run_model_on_test(config)

    cap.release()
    cv2.destroyAllWindows()
