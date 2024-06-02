import os
import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

last_path = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],
             [1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1],
             [0, 1, 1, 0, 0], [0, 1, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 1, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1],
             [0, 0, 1, 1, 1], [0, 1, 0, 1, 1], [0, 1, 1, 0, 1], [0, 1, 1, 1, 0],
             [1, 0, 0, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 1, 0], [1, 1, 0, 0, 1], [1, 1, 0, 1, 0], [1, 1, 1, 0, 0],
             [0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1],
             [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
prefix = [f"{i:03d}" for i in range(593, 625)]
prefix_to_label = dict(zip(prefix, last_path))


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


# 각각의 데이터셋에 대해 함수를 호출
train_folder = r'E:\AI\dataset_skeleton_sep\face\KneePushup\training'
valid_folder = r'E:\AI\dataset_skeleton_sep\face\KneePushup\validation'
test_folder = r'E:\AI\dataset_skeleton_sep\face\KneePushup\test'

train_image_paths, train_label_data = process_dataset(train_folder)
valid_image_paths, valid_label_data = process_dataset(valid_folder)
test_image_paths, test_label_data = process_dataset(test_folder)

# 필요에 따라 결과를 확인하거나 다른 처리를 수행
print(len(train_image_paths), len(train_label_data))
print(len(valid_image_paths), len(valid_label_data))
print(len(test_image_paths), len(test_label_data))

# ImageDataGenerator 인스턴스 생성
datagen = ImageDataGenerator(rescale=1. / 255)


def multilabel_data_generator(image_paths, label_data, batch_size):
    num_samples = len(image_paths)
    while True:  # 무한 루프로 제너레이터 구현
        for offset in range(0, num_samples, batch_size):
            batch_images = []
            batch_labels = []

            # 배치 크기만큼 이미지와 레이블 데이터 로드 및 전처리
            batch_image_paths = image_paths[offset:offset + batch_size]
            batch_image_labels = label_data[offset:offset + batch_size]

            for img_path, labels in zip(batch_image_paths, batch_image_labels):
                img = load_img(img_path, target_size=(128, 128))  # 이미지 로드 및 크기 조정
                img = img_to_array(img)  # 이미지를 numpy 배열로 변환
                img = datagen.standardize(img)  # 데이터 전처리

                batch_images.append(img)
                batch_labels.append(labels)

            # 배치 데이터 반환
            yield np.array(batch_images), np.array(batch_labels)


batch_size = 32
train_generator = multilabel_data_generator(train_image_paths, train_label_data, batch_size)
validation_generator = multilabel_data_generator(valid_image_paths, valid_label_data, batch_size)
test_generator = multilabel_data_generator(test_image_paths, test_label_data, batch_size)

# 모델 구성
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
for layer in base_model.layers:
    base_model.trainable = False
for layer in base_model.layers[-9:]:
    base_model.trainable = True

model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(5, activation='sigmoid'))

earlystopping = EarlyStopping(monitor='val_loss', patience=7, mode='min', verbose=1)

model.compile(optimizer=optimizers.Adam(learning_rate=0.0002),
              loss=['binary_crossentropy'],
              metrics=['accuracy'])

# 모델 훈련
history = model.fit(train_generator,
                    steps_per_epoch=len(train_image_paths) // batch_size,
                    epochs=25,
                    validation_data=validation_generator,
                    validation_steps=len(valid_image_paths) // batch_size,
                    #callbacks=[earlystopping]
                    )

# 테스트 데이터셋의 전체 샘플 수 계산
test_steps = np.ceil(len(test_image_paths) / batch_size)

# 모델 평가
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)

# 테스트 손실과 정확도 출력
print(f"\n테스트 손실: {test_loss}")
print(f"테스트 정확도: {test_accuracy}")

predictions = model.predict(test_generator, steps=test_steps)

# 임계값 설정
threshold = 0.5
predictions_binary = (predictions > threshold).astype(int)

# 각 레이블에 대한 정확도 계산
accuracy_per_label = np.mean(predictions_binary == test_label_data, axis=0)

# 각 레이블별 정확도 출력
for i, accuracy in enumerate(accuracy_per_label):
    print(f"레이블 {i}의 정확도: {accuracy}")

# 전체 정확도도 여전히 중요할 수 있으므로, 이를 계산합니다.
overall_accuracy = np.mean(predictions_binary == test_label_data)
print(f"전체 정확도: {overall_accuracy}")

#model.save(r'E:\AImodel\models\Face-KneePushup-multiLabel-model')