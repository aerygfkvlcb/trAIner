import os
import json
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import cv2

exercise_names01 = [
    'BicycleCrunch',              #0
    'BurpeeTest',                 #1
    'BurpeeTest',                 #2
    'CrossLunge',                 #3
    'CrossLunge',                 #4
    'Crunch',                     #5
    'Goodmorning',                #6
    'HipThrust',                  #7
    'KneePushup',                 #8
    'LyingLegRaise',              #9
    'Plank',                      #10
    'Pushup',                     #11
    'ScissorCross',               #12
    'SideLunge',                  #13
    'SideLunge',                  #14
    'StandingKneeup',             #15
    'StandingKneeup',             #16
    'StandingSideCrunch',         #17
    'StandingSideCrunch',         #18
    'StepBackwardDynamicLunge',   #19
    'StepForwardDynamicLunge',    #20
    ]

exercise_names02 = [
    'BicycleCrunch',              #0
    'BurpeeTest-face',            #1
    'BurpeeTest-side',            #2
    'CrossLunge-face',            #3
    'CrossLunge-side',            #4
    'Crunch',                     #5
    'Goodmorning',                #6 side
    'HipThrust',                  #7
    'KneePushup',                 #8
    'LyingLegRaise',              #9
    'Plank',                      #10
    'Pushup',                     #11
    'ScissorCross',               #12
    'SideLunge-face',             #13
    'SideLunge-side',             #14
    'StandingKneeup-face',        #15
    'StandingKneeup-side',        #16
    'StandingSideCrunch-face',    #17
    'StandingSideCrunch-side',    #18
    'StepBackwardDynamicLunge',   #19 side
    'StepForwardDynamicLunge',    #20 side
    ]


def resize_img(image_paths):
    images_resized = []  # 리사이즈된 이미지를 저장할 리스트
    for image_path in image_paths:
        image = cv2.imread(image_path)  # 각 이미지 경로로부터 이미지를 읽음
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 색상 변환
        image_resized = cv2.resize(image, (128, 128))  # 이미지 리사이즈
        images_resized.append(image_resized)  # 결과 리스트에 추가
    images_resized = np.array(images_resized) / 255.0  # numpy 배열로 변환 및 정규화
    return images_resized


def process_dataset(root_folder):
    image_paths = []
    label_data = []

    for roots, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.jpg'):
                # 파일 이름 분석을 위해 숫자만 추출
                prefix = file[0:3]

                # 접두사에 따른 레이블 할당
                label = prefix_to_label.get(prefix)

                # 유효한 레이블이 있는 경우에만 리스트에 추가
                if label is not None:
                    image_paths.append(os.path.join(roots, file))
                    label_data.append(label)

    return image_paths, label_data


def apply_sliding_window(sequence, window_size=5, method='mean'):
    smoothed_sequence = []
    for i in range(len(sequence)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(sequence), i + window_size // 2 + 1)
        window = sequence[start_idx:end_idx]
        if method == 'mean':
            smoothed_value = np.mean(window)
        elif method == 'median':
            smoothed_value = np.median(window)
        else:
            raise ValueError("Method should be 'mean' or 'median'")
        smoothed_sequence.append(smoothed_value)
    return smoothed_sequence


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def get_prefix_to_label(file_path, exercise_name):
    data = load_json_file(file_path)
    result = (data.get(exercise_name))
    prefix = data.get(exercise_name)['prefix']
    label = data.get(exercise_name)['last_path']
    prefix_to_label = dict(zip(prefix, label))
    return prefix_to_label


get_label_nums = lambda x: len(next(iter(x.values())))

json_path = 'E:/AInotes/자세교정/모델학습/label_data.json'
pretrained_model = 'ResNetTuning'
direction = 'face' #side
exercise_name01 = exercise_names01[0]
exercise_name02 = exercise_names02[0].lower()
test_folder = f'E:/AI/dataset_skeleton_sep/{direction}/{exercise_name01}/test'
prefix_to_label = get_prefix_to_label(json_path, exercise_name02)
#model = tf.keras.models.load_model(f'E:/AImodel/models/Multilabel/{pretrained_model}/{exercise_name02}.h5', custom_objects={'PerClassAccuracy': PerClassAccuracy})
model = tf.keras.models.load_model(f'E:/AImodel/models/Multilabel/ResNetTuning/bicyclecrunch.h5')
print(exercise_name02)

test_image_paths, test_label_data = process_dataset(test_folder)
print(len(test_image_paths), len(test_label_data))

test_image_resized = resize_img(test_image_paths)
# 모델 예측
predictions = model.predict(test_image_resized)

# 임계값 설정 (예: 0.5)
threshold = 0.5
predictions_binary = (predictions > threshold).astype(int)

# 각 레이블에 대한 정확도 계산
accuracy_per_label = np.mean(predictions_binary == test_label_data, axis=0)
precision, recall, f1_score, _ = precision_recall_fscore_support(test_label_data, predictions_binary, average=None)

print()
# 각 레이블별 정확도 출력
for i, (a,p,r,f) in enumerate(zip(accuracy_per_label, precision, recall, f1_score)):
    print(f"레이블 {i+1}의 정확도: {a:.4f}, 정밀도: {p:.4f}, 재현율: {r:.4f}, F1 스코어: {f:.4f}")

# 예측 결과를 전치(Transpose)하여 각 레이블의 모든 예측 결과를 연속으로 배열
transposed_predictions = np.transpose(predictions)

smoothed_predictions = []
for label_predictions in transposed_predictions:
    # 각 레이블 별로 예측값에 슬라이딩 윈도우 적용
    smoothed = apply_sliding_window(label_predictions, window_size=5, method='mean')
    smoothed_predictions.append(smoothed)

# 슬라이딩 윈도우 적용 결과를 다시 전치하여 원래 형태로 복원
smoothed_predictions = np.transpose(smoothed_predictions)

# Apply threshold to smoothed predictions
smoothed_predictions_binary = (smoothed_predictions > threshold).astype(int)

# Calculate accuracy for each label in smoothed predictions
smoothed_accuracy_per_label = np.mean(smoothed_predictions_binary == test_label_data, axis=0)
smoothed_precision, smoothed_recall, smoothed_f1_score, _ = precision_recall_fscore_support(test_label_data, smoothed_predictions_binary, average=None)

# Print accuracy for each label
for i, (a,p,r,f) in enumerate(zip(smoothed_accuracy_per_label, smoothed_precision, smoothed_recall, smoothed_f1_score)):
    print(f"레이블 {i+1}의 보정된 정확도: {a:.4f}, 정밀도: {p:.4f}, 재현율: {r:.4f}, F1 스코어: {f:.4f}")
