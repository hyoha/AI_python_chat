import pandas as pd
from collections import Counter
import random
import os
from sklearn.linear_model import LinearRegression
import numpy as np

# 데이터 파일 경로
file_path = 'C:/Wokrspace/PycharmProjects/pythonProject/sample/lotto.txt'

# 파일 경로 확인
if not os.path.exists(file_path):
    print(f"파일을 찾을 수 없습니다: {file_path}")
    exit(1)

# 데이터 파일을 공백으로 구분하여 읽어오기
with open(file_path, 'r') as file:
    lines = file.readlines()

# 각 라인의 공백 제거 및 숫자 리스트로 변환
data = [list(map(int, line.split())) for line in lines if line.strip()]

# 7자리 숫자에서 마지막 숫자(보너스 번호)를 제외하고 6자리로 변환
cleaned_data = [numbers[:6] if len(numbers) == 7 else numbers for numbers in data]

# 데이터가 여러 행에 걸쳐 있는 경우, 데이터 프레임을 일차원 시리즈로 변환
numbers = pd.Series([num for sublist in cleaned_data for num in sublist])

# 최근 당첨 번호 (마지막 행 기준)
last_draw = cleaned_data[-1]

# 분석 함수들 정의
def analyze_consecutive_numbers(numbers):
    consecutive_counts = []
    for subset in cleaned_data:
        subset.sort()
        count = 0
        for i in range(1, len(subset)):
            if subset[i] == subset[i - 1] + 1:
                count += 1
            else:
                if count > 0:
                    consecutive_counts.append(count + 1)
                count = 0
        if count > 0:
            consecutive_counts.append(count + 1)
    return Counter(consecutive_counts)

def analyze_number_ranges(numbers, range_size=10):
    range_counts = Counter((num // range_size) * range_size for num in numbers)
    return range_counts

def analyze_specific_numbers(numbers):
    specific_counts = Counter(numbers)
    return specific_counts

def trend_analysis():
    X = np.arange(len(cleaned_data)).reshape(-1, 1)
    y = [sum(subset) for subset in cleaned_data]
    model = LinearRegression()
    model.fit(X, y)
    next_index = np.array([[len(cleaned_data) + 1]])
    predicted_sum = model.predict(next_index)[0]
    return predicted_sum

def analyze_odd_even_ratio(numbers):
    odd_even_ratio_counts = []
    for subset in cleaned_data:
        odd_count = len([num for num in subset if num % 2 != 0])
        even_count = len([num for num in subset if num % 2 == 0])
        odd_even_ratio_counts.append((odd_count, even_count))
    return Counter(odd_even_ratio_counts)

def analyze_last_draw_similarity(prediction, last_draw):
    return len(set(prediction) & set(last_draw))

consecutive_numbers = analyze_consecutive_numbers(numbers)
sorted_consecutive_numbers = sorted(consecutive_numbers.items(), key=lambda x: x[1], reverse=True)

number_ranges = analyze_number_ranges(numbers, range_size=10)
sorted_number_ranges = sorted(number_ranges.items(), key=lambda x: x[1], reverse=True)

specific_number_counts = analyze_specific_numbers(numbers)
sorted_specific_number_counts = sorted(specific_number_counts.items(), key=lambda x: x[1], reverse=True)

predicted_sum_trend = trend_analysis()

odd_even_ratios = analyze_odd_even_ratio(numbers)
sorted_odd_even_ratios = sorted(odd_even_ratios.items(), key=lambda x: x[1], reverse=True)

# 번호 생성 함수
def generate_numbers_based_on_patterns(num_predictions=10):
    all_numbers = list(range(1, 46))
    predictions = []

    def create_prediction():
        prediction = set()
        while len(prediction) < 6:
            num = random.choice(all_numbers)
            prediction.add(num)
        return sorted(prediction)

    def is_valid_prediction(prediction):
        valid = True
        if sorted_consecutive_numbers:
            most_common_length = sorted_consecutive_numbers[0][0]
            # 연속된 2자리 숫자가 있는지 확인
            if not any(prediction[i] == prediction[i - 1] + 1 for i in range(1, len(prediction))):
                valid = False
        if sorted_number_ranges:
            most_common_range = sorted_number_ranges[0][0]
            if not any(num in range(most_common_range, most_common_range + 10) for num in prediction):
                valid = False
        return valid

    while len(predictions) < num_predictions:
        prediction = create_prediction()
        if is_valid_prediction(prediction):
            if prediction not in predictions:
                predictions.append(prediction)

    return predictions

# 확률 계산 함수 수정
def calculate_probability(prediction):
    probability = 1.0

    if sorted_consecutive_numbers:
        if 2 in dict(sorted_consecutive_numbers) and dict(sorted_consecutive_numbers)[2] >= 622:
            if any(prediction[i] == prediction[i - 1] + 1 for i in range(1, len(prediction))):
                probability *= 1.2  # 가중치 조정

    if sorted_number_ranges:
        for range_start, count in sorted_number_ranges:
            if range_start in [10, 20, 30, 0]:
                if any(num in range(range_start, range_start + 10) for num in prediction):
                    probability *= 1.1  # 가중치 조정

    if sorted_specific_number_counts:
        for num in prediction:
            if num in dict(sorted_specific_number_counts):
                probability *= (1 + dict(sorted_specific_number_counts)[num] / 1000)  # 더 작은 가중치

    if predicted_sum_trend:
        if sum(prediction) in range(int(predicted_sum_trend - 5), int(predicted_sum_trend + 5)):
            probability *= 1.1  # 가중치 조정

    similarity_to_last_draw = analyze_last_draw_similarity(prediction, last_draw)
    if similarity_to_last_draw >= 2:
        probability *= 1.2  # 가중치 조정

    if sorted_odd_even_ratios:
        most_common_ratio = sorted_odd_even_ratios[0][0]
        odd_count = len([num for num in prediction if num % 2 != 0])
        even_count = len([num for num in prediction if num % 2 == 0])
        if (odd_count, even_count) == most_common_ratio:
            probability *= 1.1  # 가중치 조정

    return min(probability, 5)  # 확률 상한 설정

predicted_numbers = generate_numbers_based_on_patterns(num_predictions=10)
print("\n예측된 로또 번호 조합과 각 조합의 예상 확률:")
for prediction in predicted_numbers:
    probability = calculate_probability(prediction)
    print(f"{prediction} - 예상 확률: {probability:.4f}")

# 결과 출력
print("\n연속된 숫자 분석 결과 (많은 순서):")
for length, count in sorted_consecutive_numbers:
    print(f"{length}개 연속된 숫자: {count}회")

print("\n번호 구간별 빈도 분석 결과 (많은 순서):")
for range_start, count in sorted_number_ranges:
    print(f"{range_start}-{range_start + 9}: {count}회")

print("\n특정 숫자 빈도 분석 결과 (많은 순서):")
for num, count in sorted_specific_number_counts:
    print(f"{num}: {count}회")

print("\n홀수/짝수 비율 분석 결과 (많은 순서):")
for (odd_count, even_count), count in sorted_odd_even_ratios:
    print(f"홀수 {odd_count}, 짝수 {even_count}: {count}회")
