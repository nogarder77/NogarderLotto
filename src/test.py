"""
로또 번호 예측 프로그램 단위 테스트
"""
import unittest
import os
import sys
import numpy as np
import pandas as pd

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collector import LottoDataCollector
from data.preprocessor import DataPreprocessor
from model.cdm_model import CDMModel
from model.number_generator import NumberGenerator
from utils.helpers import validate_lotto_numbers


class TestLottoDataCollector(unittest.TestCase):
    """LottoDataCollector 테스트"""
    
    def test_fetch_draw(self):
        """단일 회차 데이터 가져오기 테스트"""
        collector = LottoDataCollector()
        data = collector._fetch_draw(1)  # 1회차 데이터
        
        self.assertIsNotNone(data)
        self.assertEqual(data.get("returnValue"), "success")
        self.assertEqual(data.get("drwNo"), 1)
        self.assertIsNotNone(data.get("drwtNo1"))
    
    def test_find_latest_draw(self):
        """최신 회차 찾기 테스트"""
        collector = LottoDataCollector()
        latest = collector._find_latest_draw()
        
        self.assertIsNotNone(latest)
        self.assertGreater(latest, 1000)  # 현재 1000회차 이상임을 가정


class TestDataPreprocessor(unittest.TestCase):
    """DataPreprocessor 테스트"""
    
    def setUp(self):
        """테스트용 데이터 생성"""
        self.test_data = pd.DataFrame({
            'draw_no': [1, 2, 3],
            'date': ['2002-12-07', '2002-12-14', '2002-12-21'],
            'num1': [10, 9, 11],
            'num2': [23, 13, 19],
            'num3': [29, 21, 25],
            'num4': [33, 25, 27],
            'num5': [37, 32, 30],
            'num6': [40, 42, 38],
            'bonus': [16, 2, 15]
        })
    
    def test_load_data(self):
        """데이터 로드 테스트"""
        preprocessor = DataPreprocessor()
        data = preprocessor.load_data(self.test_data)
        
        self.assertEqual(len(data), 3)
    
    def test_get_number_combinations(self):
        """번호 조합 추출 테스트"""
        preprocessor = DataPreprocessor(self.test_data)
        combinations = preprocessor.get_number_combinations()
        
        self.assertEqual(len(combinations), 3)
        self.assertEqual(len(combinations[0]), 6)


class TestCDMModel(unittest.TestCase):
    """CDMModel 테스트"""
    
    def test_fit_and_generate(self):
        """모델 학습 및 번호 생성 테스트"""
        # 테스트 데이터
        data = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18]
        ]
        
        # 모델 초기화 및 학습
        model = CDMModel(seed=42)
        model.fit(data)
        
        # 번호 생성
        numbers = model.generate_numbers()
        
        self.assertEqual(len(numbers), 6)
        self.assertTrue(all(1 <= n <= 45 for n in numbers))
        self.assertEqual(len(set(numbers)), 6)  # 중복 없음
    
    def test_probability_calculation(self):
        """확률 계산 테스트"""
        # 테스트 데이터
        data = [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 7, 8, 9],
            [1, 2, 10, 11, 12, 13]
        ]
        
        # 모델 초기화 및 학습
        model = CDMModel(seed=42)
        model.fit(data)
        
        # 자주 등장한 번호 조합과 그렇지 않은 조합 비교
        prob1 = model.calculate_combination_probability([1, 2, 3, 4, 5, 6])
        prob2 = model.calculate_combination_probability([40, 41, 42, 43, 44, 45])
        
        # 자주 등장한 번호 조합이 더 높은 확률을 가져야 함
        self.assertGreater(prob1, prob2)


class TestNumberGenerator(unittest.TestCase):
    """NumberGenerator 테스트"""
    
    def test_random_sampling(self):
        """무작위 샘플링 테스트"""
        generator = NumberGenerator(seed=42)
        numbers = generator.random_sampling()
        
        self.assertEqual(len(numbers), 6)
        self.assertTrue(all(1 <= n <= 45 for n in numbers))
        self.assertEqual(len(set(numbers)), 6)  # 중복 없음
    
    def test_generate_with_cdm(self):
        """CDM 기반 샘플링 테스트"""
        # 테스트 데이터
        data = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18]
        ]
        
        # CDM 모델 설정
        model = CDMModel(seed=42)
        model.fit(data)
        
        # 번호 생성기
        generator = NumberGenerator(cdm_model=model, seed=42)
        numbers = generator.generate_with_cdm()
        
        self.assertEqual(len(numbers), 6)
        self.assertTrue(all(1 <= n <= 45 for n in numbers))
        self.assertEqual(len(set(numbers)), 6)  # 중복 없음


class TestHelpers(unittest.TestCase):
    """유틸리티 함수 테스트"""
    
    def test_validate_lotto_numbers(self):
        """번호 유효성 검증 테스트"""
        # 유효한 번호
        self.assertTrue(validate_lotto_numbers([1, 2, 3, 4, 5, 6]))
        
        # 잘못된 개수
        self.assertFalse(validate_lotto_numbers([1, 2, 3, 4, 5]))
        
        # 범위 벗어남
        self.assertFalse(validate_lotto_numbers([0, 1, 2, 3, 4, 5]))
        self.assertFalse(validate_lotto_numbers([1, 2, 3, 4, 5, 46]))
        
        # 중복된 번호
        self.assertFalse(validate_lotto_numbers([1, 2, 3, 4, 5, 5]))


if __name__ == "__main__":
    unittest.main() 