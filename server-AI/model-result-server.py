import os
import sys
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

exercise_name = (sys.argv[1]).lower()
image_folder_path = sys.argv[2]
json_path = 'modelLink/label_data.json'  # 심볼릭 링크로 .py 파일 있는 곳에 json 데이터 링크 배치
model = (tf.keras.models.load_model(f'modelLink/{exercise_name}.h5'))  # 심볼릭 링크로 .py 파일 있는 곳에 모델 링크 배치


def preprocess_image(image_path, target_size=(128, 128)):  # 이미지 파일을 불러와 전처리하는 함수
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 모델 예측을 위해 차원 확장
    img_array /= 255.0  # 이미지를 0과 1 사이로 스케일링
    return img_array


def load_and_preprocess_from_directory(directory_path, target_size=(128, 128)):  # 지정된 디렉토리 내의 모든 이미지를 불러와 전처리하는 함수
    processed_images = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = preprocess_image(file_path, target_size=target_size)
            processed_images.append(img)
    return np.vstack(processed_images)  # 전처리된 이미지들을 하나의 numpy 배열로 합침


def apply_sliding_window_to_predictions(predictions, window_size=5, method='mean'):  # 2D 예측 결과 배열에 대해 슬라이딩 윈도우를 적용하는 함수
    num_classes = predictions.shape[1]  # 클래스 수
    smoothed_predictions = np.zeros_like(predictions)

    for class_idx in range(num_classes):
        class_predictions = predictions[:, class_idx]  # 현재 클래스에 대한 모든 예측값
        # 각 예측값에 대해 슬라이딩 윈도우 적용
        for i in range(len(class_predictions)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(class_predictions), i + window_size // 2 + 1)
            window = class_predictions[start_idx:end_idx]
            if method == 'mean':
                smoothed_value = np.mean(window)
            elif method == 'median':
                smoothed_value = np.median(window)
            else:
                raise ValueError("Method should be 'mean' or 'median'")
            smoothed_predictions[i, class_idx] = smoothed_value

    return smoothed_predictions


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def get_prefix_to_label(file_path, exercise_name):
    data = load_json_file(file_path)
    prefix = data.get(exercise_name)['prefix']
    label = data.get(exercise_name)['last_path']
    prefix_to_label = dict(zip(prefix, label))
    return prefix_to_label


def print_predictions(predictions, exercise_name, json_path, threshold=0.5):
    # JSON 파일 로드 및 라벨 길이 구하기
    data = load_json_file(json_path)
    label_length = len(data[exercise_name]['last_path'][0])

    # 각 라벨마다 T, F의 개수를 저장할 딕셔너리 동적으로 초기화
    label_counts = {i + 1: {'T': 0, 'F': 0} for i in range(label_length)}

    for i, prediction in enumerate(predictions):
        labels = (prediction > threshold).astype(int)
        # 각 라벨마다 T, F의 개수를 계산
        for label_index, label in enumerate(labels, start=1):
            if label == 1:
                label_counts[label_index]['T'] += 1
            else:
                label_counts[label_index]['F'] += 1
# 각 라벨에 대해 T, F 중 더 많은 것을 출력
    result = []
    for label_index in label_counts:
        more_common = 'T' if label_counts[label_index]['T'] > label_counts[label_index]['F'] else 'F'
        result.append(more_common)
    print(result)


preprocessed_images = load_and_preprocess_from_directory(image_folder_path)
predictions = model.predict(preprocessed_images, verbose=0)
smoothed_prediction = apply_sliding_window_to_predictions(predictions, window_size=5, method='mean')
# 예측 결과 출력
print_predictions(smoothed_prediction, exercise_name, json_path)
