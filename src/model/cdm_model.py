"""
Compound-Dirichlet-Multinomial (CDM) 모델 구현
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns


class CDMModel:
    """Compound-Dirichlet-Multinomial (CDM) 모델 클래스"""
    
    def __init__(self, 
                 prior_alpha: Optional[np.ndarray] = None,
                 n_numbers: int = 45,
                 seed: Optional[int] = None):
        """
        CDM 모델 초기화
        
        Args:
            prior_alpha: 사전 분포 파라미터 (Dirichlet 분포의 알파값), 
                        기본값은 모든 번호에 대해 1.0 (균등 분포)
            n_numbers: 로또 번호 범위 (기본값: 45)
            seed: 난수 생성 시드
        """
        self.n_numbers = n_numbers
        
        # 기본 사전 분포 설정 (균등 분포)
        if prior_alpha is None:
            self.prior_alpha = np.ones(n_numbers)
        else:
            if len(prior_alpha) != n_numbers:
                raise ValueError(f"prior_alpha 길이는 n_numbers({n_numbers})와 같아야 합니다.")
            self.prior_alpha = prior_alpha
            
        # 학습 후 사후 분포 파라미터 저장
        self.posterior_alpha = None
        
        # 난수 생성기 설정
        self.rng = np.random.RandomState(seed)
    
    def fit(self, data: Union[List[List[int]], np.ndarray, pd.DataFrame], 
            column_names: Optional[List[str]] = None) -> None:
        """
        CDM 모델 학습 (사후 분포 계산)
        
        Args:
            data: 로또 당첨 번호 데이터
                - List[List[int]]: 각 행이 6개의 당첨번호 리스트
                - np.ndarray: 각 행이 6개의 당첨번호 배열
                - pd.DataFrame: 각 행이 로또 회차, 열이 번호
            column_names: DataFrame 경우, 번호가 포함된 열 이름들
        """
        # 데이터 형식에 따른 처리
        if isinstance(data, pd.DataFrame):
            if column_names is None:
                # 기본 열 이름 (num1 ~ num6)
                column_names = [f'num{i}' for i in range(1, 7)]
            
            # DataFrame에서 당첨번호 추출
            num_data = data[column_names].values
            # 결측값 처리
            num_data = np.array([[int(n) for n in row if not np.isnan(n)] for row in num_data])
        else:
            # 리스트나 배열인 경우 그대로 사용
            num_data = data
        
        # 번호를 0-indexed로 변환 (1~45 -> 0~44)
        num_data_0idx = [[n-1 for n in row] for row in num_data]
        
        # 빈도 계산
        frequency = np.zeros(self.n_numbers)
        for row in num_data_0idx:
            for num in row:
                frequency[num] += 1
        
        # 사후 분포 계산 (사전 분포 + 관측 빈도)
        self.posterior_alpha = self.prior_alpha + frequency
        
    def sample_probability(self) -> np.ndarray:
        """
        사후 분포에서 확률 벡터 샘플링
        
        Returns:
            번호별 확률 벡터 (길이: n_numbers)
        """
        if self.posterior_alpha is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        # Dirichlet 분포에서 확률 벡터 샘플링
        return self.rng.dirichlet(self.posterior_alpha)
    
    def generate_numbers(self, n_samples: int = 6) -> List[int]:
        """
        로또 번호 조합 생성
        
        Args:
            n_samples: 생성할 번호 개수 (기본값: 6)
            
        Returns:
            생성된 번호 리스트 (1~45 범위)
        """
        if self.posterior_alpha is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        # 확률 벡터 샘플링
        prob = self.sample_probability()
        
        # 확률에 따라 번호 추출 (복원추출 없이)
        # 0부터 시작하는 인덱스로 샘플링 후 1을 더해 1~45 범위로 변환
        sampled_indices = self.rng.choice(
            self.n_numbers, size=n_samples, replace=False, p=prob
        )
        
        # 1부터 시작하는 번호로 변환 및 정렬
        return sorted([idx + 1 for idx in sampled_indices])
    
    def generate_multiple_combinations(self, n_combinations: int = 5, n_numbers: int = 6) -> List[List[int]]:
        """
        여러 개의 번호 조합 생성
        
        Args:
            n_combinations: 생성할 조합 개수 (기본값: 5)
            n_numbers: 각 조합의 번호 개수 (기본값: 6)
            
        Returns:
            생성된 번호 조합 리스트
        """
        combinations = []
        
        for _ in range(n_combinations):
            combination = self.generate_numbers(n_numbers)
            combinations.append(combination)
            
        return combinations
    
    def calculate_combination_probability(self, combination: List[int]) -> float:
        """
        특정 번호 조합의 확률 계산
        
        Args:
            combination: 확률을 계산할 번호 조합 (1~45 범위)
            
        Returns:
            조합의 확률 값
        """
        if self.posterior_alpha is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        # 0-indexed로 변환
        indices = [num - 1 for num in combination]
        
        # 조합 확률 계산을 위한 Monte Carlo 샘플링
        n_samples = 10000
        probability_sum = 0.0
        
        for _ in range(n_samples):
            prob_vector = self.sample_probability()
            # 복원 추출 없는 확률 계산 (다항 계수 고려)
            unnormalized_prob = np.prod(prob_vector[indices])
            probability_sum += unnormalized_prob
        
        return probability_sum / n_samples
    
    def generate_and_score_combinations(self, 
                                       n_combinations: int = 20, 
                                       n_to_select: int = 5,
                                       n_numbers: int = 6) -> List[Dict[str, Any]]:
        """
        번호 조합 생성 및 점수화
        
        Args:
            n_combinations: 생성할 초기 조합 개수 (기본값: 20)
            n_to_select: 최종 선택할 조합 개수 (기본값: 5)
            n_numbers: 각 조합의 번호 개수 (기본값: 6)
            
        Returns:
            생성된 번호 조합 및 점수 정보
        """
        all_combinations = self.generate_multiple_combinations(n_combinations, n_numbers)
        scored_combinations = []
        
        for combination in all_combinations:
            # 조합 확률 계산
            probability = self.calculate_combination_probability(combination)
            
            # 결과 저장
            scored_combinations.append({
                'numbers': combination,
                'probability': probability,
                'confidence': probability * 100  # 백분율로 변환
            })
        
        # 확률에 따라 정렬
        scored_combinations.sort(key=lambda x: x['probability'], reverse=True)
        
        # 상위 n_to_select개 조합 선택
        return scored_combinations[:n_to_select]
    
    def get_number_probabilities(self) -> pd.DataFrame:
        """
        각 번호의 사후 확률 계산
        
        Returns:
            번호별 사후 확률 DataFrame
        """
        if self.posterior_alpha is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        # 사후 분포 파라미터 합계
        alpha_sum = np.sum(self.posterior_alpha)
        
        # 각 번호의 기대 확률 계산
        expected_probs = self.posterior_alpha / alpha_sum
        
        # 결과 DataFrame 생성
        prob_df = pd.DataFrame({
            'number': range(1, self.n_numbers + 1),
            'prior_alpha': self.prior_alpha,
            'posterior_alpha': self.posterior_alpha,
            'expected_probability': expected_probs
        })
        
        return prob_df
    
    def plot_number_probabilities(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        번호별 사후 확률 시각화
        
        Args:
            figsize: 그래프 크기
            
        Returns:
            Matplotlib Figure 객체
        """
        if self.posterior_alpha is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        # 번호별 확률 데이터 가져오기
        prob_df = self.get_number_probabilities()
        
        # 시각화
        fig, ax = plt.subplots(figsize=figsize)
        
        # 막대 그래프
        sns.barplot(x='number', y='expected_probability', data=prob_df, ax=ax)
        
        ax.set_title('로또 번호별 예측 확률')
        ax.set_xlabel('번호')
        ax.set_ylabel('확률')
        
        # x축 레이블 설정 (5의 배수만 표시)
        ax.set_xticks([i for i in range(0, self.n_numbers, 5)])
        ax.set_xticklabels([i+1 for i in range(0, self.n_numbers, 5)])
        
        plt.tight_layout()
        return fig 