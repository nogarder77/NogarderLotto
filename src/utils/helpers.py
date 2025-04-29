"""
유틸리티 함수 모듈
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple


def validate_lotto_numbers(numbers: List[int]) -> bool:
    """
    로또 번호 유효성 검증
    
    Args:
        numbers: 검증할 번호 리스트
        
    Returns:
        유효성 여부 (True/False)
    """
    # 6개의 번호인지 확인
    if len(numbers) != 6:
        return False
    
    # 범위 검증 (1~45)
    if not all(1 <= num <= 45 for num in numbers):
        return False
    
    # 중복 없는지 확인
    if len(set(numbers)) != 6:
        return False
    
    return True


def calculate_win_statistics(draw_data: pd.DataFrame) -> Dict[str, Any]:
    """
    당첨 통계 계산
    
    Args:
        draw_data: 당첨 데이터 DataFrame
        
    Returns:
        통계 정보 딕셔너리
    """
    stats = {}
    
    # 번호별 출현 빈도
    number_counts = {}
    for i in range(1, 7):
        col = f'num{i}'
        for num in draw_data[col]:
            if np.isnan(num):
                continue
            num = int(num)
            number_counts[num] = number_counts.get(num, 0) + 1
    
    # 가장 많이 나온 번호
    most_frequent = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # 가장 적게 나온 번호
    least_frequent = sorted(number_counts.items(), key=lambda x: x[1])[:10]
    
    # 번호별 출현 간격
    gaps = {}
    for num in range(1, 46):
        appearances = []
        for i, row in draw_data.iterrows():
            if any(row[f'num{j}'] == num for j in range(1, 7) if not np.isnan(row[f'num{j}'])):
                appearances.append(row['draw_no'])
        
        if len(appearances) > 1:
            # 연속 회차 간 간격 계산
            num_gaps = [appearances[i] - appearances[i-1] for i in range(1, len(appearances))]
            gaps[num] = {
                'avg_gap': np.mean(num_gaps),
                'max_gap': max(num_gaps),
                'min_gap': min(num_gaps),
                'last_appearance': appearances[-1],
                'total_appearances': len(appearances)
            }
    
    # 결과 저장
    stats['most_frequent'] = most_frequent
    stats['least_frequent'] = least_frequent
    stats['gaps'] = gaps
    
    return stats


def check_combination_overlap(new_combination: List[int], 
                            existing_combinations: List[List[int]],
                            min_overlap: int = 4) -> bool:
    """
    새 조합이 기존 조합들과 많이 겹치는지 확인
    
    Args:
        new_combination: 새 번호 조합
        existing_combinations: 기존 번호 조합 목록
        min_overlap: 겹침으로 간주할 최소 일치 개수
        
    Returns:
        많이 겹치면 True, 아니면 False
    """
    new_set = set(new_combination)
    
    for combo in existing_combinations:
        existing_set = set(combo)
        overlap_count = len(new_set.intersection(existing_set))
        
        if overlap_count >= min_overlap:
            return True
    
    return False


def generate_diverse_combinations(generator_func, 
                                 n_combinations: int = 5, 
                                 max_attempts: int = 100,
                                 min_overlap: int = 3) -> List[List[int]]:
    """
    다양성 있는 번호 조합 생성
    
    Args:
        generator_func: 번호 조합 생성 함수
        n_combinations: 생성할 조합 개수
        max_attempts: 최대 시도 횟수
        min_overlap: 겹침으로 간주할 최소 일치 개수
        
    Returns:
        생성된 번호 조합 리스트
    """
    combinations = []
    attempts = 0
    
    while len(combinations) < n_combinations and attempts < max_attempts:
        new_combo = generator_func()
        
        # 기존 조합들과 겹치는지 확인
        if not check_combination_overlap(new_combo, combinations, min_overlap):
            combinations.append(new_combo)
        
        attempts += 1
    
    return combinations


def save_combination_history(combinations: List[Dict[str, Any]], 
                           history_file: str = "prediction_history.csv") -> None:
    """
    번호 조합 예측 히스토리 저장
    
    Args:
        combinations: 저장할 번호 조합 리스트
        history_file: 히스토리 파일 경로
    """
    # 데이터 준비
    date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    records = []
    for i, combo in enumerate(combinations, 1):
        numbers = combo['numbers']
        confidence = combo.get('confidence', 0)
        
        record = {
            'date': date,
            'game_num': i,
            'numbers': str(numbers),
            'confidence': confidence
        }
        
        # 개별 번호도 저장
        for j, num in enumerate(sorted(numbers), 1):
            record[f'num{j}'] = num
            
        records.append(record)
    
    # DataFrame 생성
    df = pd.DataFrame(records)
    
    # 파일이 이미 존재하는지 확인
    if os.path.exists(history_file):
        # 기존 파일에 추가
        existing_df = pd.read_csv(history_file)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(history_file, index=False)
    else:
        # 새 파일 생성
        df.to_csv(history_file, index=False)
        
    print(f"예측 히스토리가 {history_file}에 저장되었습니다.") 