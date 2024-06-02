import json

json_path = 'modelLink/label_data.json'


# JSON 파일 로드하기
with open(json_path, 'r') as file:
    data = json.load(file)

# 'bicyclecrunch'의 'prefix' 리스트를 출력하기
print(data['bicyclecrunch']['prefix'])
