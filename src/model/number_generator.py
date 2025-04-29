"""
로또 번호 생성 모듈
"""
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Any, Tuple
from .cdm_model import CDMModel


class NumberGenerator:
    """로또 번호 생성 클래스"""
    
    def __init__(self,
                cdm_model: Optional[CDMModel] = None,
                min_number: int = 1,
                max_number: int = 45,
                seed: Optional[int] = None):
        """
        NumberGenerator 초기화
        
        Args:
            cdm_model: 학습된 CDM 모델 (기본값: None)
            min_number: 최소 번호 (기본값: 1)
            max_number: 최대 번호 (기본값: 45)
            seed: 난수 생성 시드
        """
        self.min_number = min_number
        self.max_number = max_number
        self.cdm_model = cdm_model
        self.n_numbers = max_number - min_number + 1
        
        # 난수 생성기 설정
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def set_model(self, model: CDMModel) -> None:
        """
        CDM 모델 설정
        
        Args:
            model: 설정할 CDM 모델
        """
        self.cdm_model = model
    
    def random_sampling(self, n_samples: int = 6) -> List[int]:
        """
        단순 무작위 샘플링
        
        Args:
            n_samples: 샘플 개수 (기본값: 6)
            
        Returns:
            생성된 번호 리스트
        """
        numbers = random.sample(range(self.min_number, self.max_number + 1), n_samples)
        return sorted(numbers)
    
    def generate_with_cdm(self, n_samples: int = 6) -> List[int]:
        """
        CDM 모델 기반 샘플링
        
        Args:
            n_samples: 샘플 개수 (기본값: 6)
            
        Returns:
            생성된 번호 리스트
        """
        if self.cdm_model is None:
            raise ValueError("CDM 모델이 설정되지 않았습니다.")
        
        return self.cdm_model.generate_numbers(n_samples)
    
    def generate_multiple_combinations(self, 
                                      n_combinations: int = 5, 
                                      n_numbers: int = 6, 
                                      method: str = 'cdm') -> List[List[int]]:
        """
        여러 개의 번호 조합 생성
        
        Args:
            n_combinations: 생성할 조합 개수 (기본값: 5)
            n_numbers: 각 조합의 번호 개수 (기본값: 6)
            method: 생성 방법 ('cdm' 또는 'random')
            
        Returns:
            생성된 번호 조합 리스트
        """
        combinations = []
        
        for _ in range(n_combinations):
            if method.lower() == 'cdm':
                numbers = self.generate_with_cdm(n_numbers)
            else:
                numbers = self.random_sampling(n_numbers)
            
            combinations.append(numbers)
        
        return combinations
    
    def score_combinations(self, combinations: List[List[int]]) -> List[Dict[str, Any]]:
        """
        번호 조합에 점수 부여
        
        Args:
            combinations: 평가할 번호 조합 리스트
            
        Returns:
            점수가 부여된 조합 정보 리스트
        """
        if self.cdm_model is None:
            raise ValueError("CDM 모델이 설정되지 않았습니다.")
        
        scored_combinations = []
        
        for combination in combinations:
            # CDM 모델을 사용한 확률 계산
            probability = self.cdm_model.calculate_combination_probability(combination)
            
            # 점수 정보 저장
            scored_combinations.append({
                'numbers': combination,
                'probability': probability,
                'confidence': probability * 100  # 백분율로 변환
            })
        
        # 확률 기준 내림차순 정렬
        scored_combinations.sort(key=lambda x: x['probability'], reverse=True)
        
        return scored_combinations
    
    def filter_combinations(self, 
                          combinations: List[List[int]], 
                          min_sum: int = 100, 
                          max_sum: int = 170, 
                          min_odd: int = 1, 
                          max_odd: int = 5) -> List[List[int]]:
        """
        통계적 특성에 따라 번호 조합 필터링
        
        Args:
            combinations: 필터링할 번호 조합 리스트
            min_sum: 최소 합계
            max_sum: 최대 합계
            min_odd: 최소 홀수 개수
            max_odd: 최대 홀수 개수
            
        Returns:
            필터링된 번호 조합 리스트
        """
        filtered = []
        
        for combo in combinations:
            # 합계 확인
            combo_sum = sum(combo)
            if not (min_sum <= combo_sum <= max_sum):
                continue
            
            # 홀수 개수 확인
            odd_count = sum(1 for num in combo if num % 2 == 1)
            if not (min_odd <= odd_count <= max_odd):
                continue
            
            # 필터 통과
            filtered.append(combo)
        
        return filtered
    
    def generate_best_combinations(self, 
                                  n_combinations: int = 20, 
                                  n_to_select: int = 5,
                                  n_numbers: int = 6,
                                  apply_filters: bool = True) -> List[Dict[str, Any]]:
        """
        최적의 번호 조합 생성
        
        Args:
            n_combinations: 초기 생성할 조합 개수 (기본값: 20)
            n_to_select: 최종 선택할 조합 개수 (기본값: 5)
            n_numbers: 각 조합의 번호 개수 (기본값: 6)
            apply_filters: 필터 적용 여부 (기본값: True)
            
        Returns:
            선택된 번호 조합 및 점수 정보
        """
        if self.cdm_model is None:
            raise ValueError("CDM 모델이 설정되지 않았습니다.")
        
        # 초기 조합 생성 (필요한 경우 더 많이 생성)
        initial_count = n_combinations * 2 if apply_filters else n_combinations
        combinations = self.cdm_model.generate_multiple_combinations(initial_count, n_numbers)
        
        # 필터 적용 (선택적)
        if apply_filters:
            combinations = self.filter_combinations(combinations)
            
            # 필터링 후 조합이 부족한 경우 추가 생성
            while len(combinations) < n_combinations:
                additional = self.cdm_model.generate_multiple_combinations(5, n_numbers)
                filtered_additional = self.filter_combinations(additional)
                combinations.extend(filtered_additional)
                
                # 무한 루프 방지
                if len(combinations) == 0:
                    print("경고: 필터에 맞는 조합을 찾을 수 없습니다. 필터를 완화합니다.")
                    combinations = self.cdm_model.generate_multiple_combinations(n_combinations, n_numbers)
                    break
        
        # 필요한 만큼만 사용
        combinations = combinations[:n_combinations]
        
        # 점수 계산 및 선택
        scored_combinations = self.score_combinations(combinations)
        
        return scored_combinations[:n_to_select]
    
    def format_output(self, combinations: List[Dict[str, Any]], start_draw: int, end_draw: int) -> str:
        """
        결과 출력 형식 지정
        
        Args:
            combinations: 점수화된 번호 조합 리스트
            start_draw: 분석 시작 회차
            end_draw: 분석 종료 회차
            
        Returns:
            형식화된 출력 문자열
        """
        output = []
        output.append("========== 로또 번호 예측 결과 ==========")
        output.append(f"[분석 기준] {start_draw}회차 ~ {end_draw}회차 데이터 기반\n")
        
        for i, combo in enumerate(combinations, 1):
            numbers = combo['numbers']
            confidence = combo['confidence']
            output.append(f"게임 {i}: {numbers} (신뢰도: {confidence:.1f}%)")
        
        output.append("\n[참고] 본 예측 결과는 확률적 모델에 기반한 것으로,")
        output.append("       당첨을 보장하지 않습니다.")
        
        return "\n".join(output) 