import pickle

test_keypoints_path = "wind2.test"

with open(test_keypoints_path, "rb") as f:
    keypoints_data = pickle.load(f)

print(type(keypoints_data))
print("Số mẫu:", len(keypoints_data))
print("Kiểu dữ liệu:", type(keypoints_data))
print(" Các keys có trong dict:")
for idx, key in enumerate(keypoints_data.keys()):
    print(f"{idx}: {key}")
