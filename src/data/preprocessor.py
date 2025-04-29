"""
로또 당첨 데이터 전처리 모듈
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any


class DataPreprocessor:
    """로또 당첨 데이터 전처리 클래스"""
    
    def __init__(self, data: Optional[Union[pd.DataFrame, str]] = None):
        """
        DataPreprocessor 초기화
        
        Args:
            data: 입력 데이터 (DataFrame 또는 CSV 파일 경로)
        """
        self.data = None
        
        if data is not None:
            self.load_data(data)
    
    def load_data(self, data: Union[pd.DataFrame, str]) -> pd.DataFrame:
        """
        데이터 로드
        
        Args:
            data: 입력 데이터 (DataFrame 또는 CSV 파일 경로)
            
        Returns:
            처리된 DataFrame
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        elif isinstance(data, str) and os.path.isfile(data):
            self.data = pd.read_csv(data)
        else:
            raise ValueError("유효한 DataFrame 또는 CSV 파일 경로를 제공해야 합니다.")
        
        # 기본 데이터 검증
        self._validate_data()
        
        return self.data
    
    def _validate_data(self) -> None:
        """데이터 유효성 검증 및 기본 정제"""
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        # 필수 칼럼 확인
        required_columns = [
            "draw_no", "date", 
            "num1", "num2", "num3", "num4", "num5", "num6", "bonus"
        ]
        
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"데이터에 필수 칼럼이 누락되었습니다: {missing_columns}")
        
        # 데이터 타입 변환
        for col in ["num1", "num2", "num3", "num4", "num5", "num6", "bonus"]:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        self.data["draw_no"] = pd.to_numeric(self.data["draw_no"], errors='coerce')
        
        # 날짜 형식 변환
        try:
            self.data["date"] = pd.to_datetime(self.data["date"])
        except:
            print("날짜 형식 변환에 실패했습니다. 원본 형식을 유지합니다.")
        
        # 결측치 확인 및 처리
        missing_values = self.data[required_columns].isnull().sum()
        if missing_values.sum() > 0:
            print(f"결측치 발견: \n{missing_values[missing_values > 0]}")
            # 중요 칼럼에 결측치가 있는 행 제거
            self.data = self.data.dropna(subset=required_columns)
            print(f"결측치가 있는 {missing_values.sum()}개 행을 제거했습니다. 남은 데이터: {len(self.data)}행")
    
    def create_number_frequency_matrix(self) -> pd.DataFrame:
        """
        각 회차별 번호 출현 빈도 행렬 생성
        
        Returns:
            회차별 번호 출현 빈도 DataFrame (행: 회차, 열: 번호 1~45)
        """
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        # 1부터 45까지의 숫자에 대한 출현 빈도 행렬 생성
        frequency_matrix = pd.DataFrame(0, 
                                        index=self.data['draw_no'], 
                                        columns=range(1, 46))
        
        # 각 회차의 당첨 번호를 행렬에 표시
        for _, row in self.data.iterrows():
            draw_idx = row['draw_no']
            for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']:
                num = row[col]
                if pd.notna(num) and 1 <= num <= 45:
                    frequency_matrix.loc[draw_idx, num] = 1
        
        return frequency_matrix
    
    def create_cumulative_frequency(self) -> pd.DataFrame:
        """
        누적 번호 출현 빈도 계산
        
        Returns:
            각 번호의 누적 출현 빈도 DataFrame
        """
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        # 각 번호의 빈도 계산
        number_columns = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        all_numbers = self.data[number_columns].values.flatten()
        all_numbers = all_numbers[~np.isnan(all_numbers)].astype(int)
        
        # 번호별 빈도 계산
        number_counts = pd.Series(all_numbers).value_counts().sort_index()
        
        # 1~45까지 모든 번호에 대한 빈도 DataFrame 생성
        frequency_df = pd.DataFrame({
            'number': range(1, 46),
            'frequency': [number_counts.get(i, 0) for i in range(1, 46)]
        })
        
        # 출현 확률 추가
        total_draws = len(self.data)
        frequency_df['probability'] = frequency_df['frequency'] / (total_draws * 6)
        
        return frequency_df
    
    def get_number_combinations(self) -> List[List[int]]:
        """
        모든 회차의 번호 조합 리스트 추출
        
        Returns:
            번호 조합 리스트 (각 항목은 6개 번호의 리스트)
        """
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        combinations = []
        for _, row in self.data.iterrows():
            numbers = [row[f'num{i}'] for i in range(1, 7)]
            numbers = [int(n) for n in numbers if pd.notna(n)]
            if len(numbers) == 6:  # 유효한 조합만 추가
                combinations.append(sorted(numbers))
        
        return combinations
    
    def analyze_number_patterns(self) -> Dict[str, Any]:
        """
        당첨 번호 패턴 분석
        
        Returns:
            다양한 통계 및 패턴 정보를 담은 딕셔너리
        """
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        combinations = self.get_number_combinations()
        
        # 번호별 출현 빈도
        frequency_df = self.create_cumulative_frequency()
        
        # 홀수/짝수 비율 분석
        odd_even_ratios = []
        for combo in combinations:
            odds = sum(1 for num in combo if num % 2 == 1)
            evens = 6 - odds
            odd_even_ratios.append((odds, evens))
        
        # 번호 구간별 분포 (1-10, 11-20, 21-30, 31-40, 41-45)
        range_distribution = []
        for combo in combinations:
            ranges = [0] * 5  # 5개 구간
            for num in combo:
                if 1 <= num <= 10:
                    ranges[0] += 1
                elif 11 <= num <= 20:
                    ranges[1] += 1
                elif 21 <= num <= 30:
                    ranges[2] += 1
                elif 31 <= num <= 40:
                    ranges[3] += 1
                elif 41 <= num <= 45:
                    ranges[4] += 1
            range_distribution.append(ranges)
        
        # 연속 번호 분석
        consecutive_counts = []
        for combo in combinations:
            sorted_combo = sorted(combo)
            consecutive_count = 0
            for i in range(len(sorted_combo) - 1):
                if sorted_combo[i + 1] - sorted_combo[i] == 1:
                    consecutive_count += 1
            consecutive_counts.append(consecutive_count)
        
        # 번호 간 간격 분석
        gaps = []
        for combo in combinations:
            sorted_combo = sorted(combo)
            combo_gaps = [sorted_combo[i+1] - sorted_combo[i] for i in range(len(sorted_combo)-1)]
            gaps.append(combo_gaps)
        
        # 합계 및 평균 분석
        sums = [sum(combo) for combo in combinations]
        
        # 결과 집계
        analysis = {
            'frequency': frequency_df,
            'odd_even_ratios': pd.Series(odd_even_ratios).value_counts().sort_index(),
            'range_distribution': np.mean(range_distribution, axis=0),
            'consecutive_counts': pd.Series(consecutive_counts).value_counts().sort_index(),
            'avg_gaps': np.mean([gap for combo_gaps in gaps for gap in combo_gaps]),
            'sum_stats': {
                'min': min(sums),
                'max': max(sums),
                'mean': np.mean(sums),
                'median': np.median(sums),
                'most_common': pd.Series(sums).value_counts().sort_values(ascending=False).head(5)
            }
        }
        
        return analysis
    
    def create_training_dataset(self, window_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        시계열 학습 데이터셋 생성
        
        Args:
            window_size: 과거 회차 윈도우 크기
            
        Returns:
            (X, y) 형태의 학습 데이터셋
            X: 과거 window_size 회차의 당첨번호 데이터
            y: 다음 회차 당첨번호
        """
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        # 번호 조합 추출
        combinations = self.get_number_combinations()
        
        # 당첨번호를 1-hot 인코딩으로 변환 (1~45 범위)
        encoded_data = np.zeros((len(combinations), 45))
        for i, combo in enumerate(combinations):
            for num in combo:
                encoded_data[i, num-1] = 1
        
        # 시계열 학습 데이터 생성
        X, y = [], []
        for i in range(len(encoded_data) - window_size):
            X.append(encoded_data[i:i+window_size])
            y.append(encoded_data[i+window_size])
        
        return np.array(X), np.array(y) 