"""
로또 당첨 데이터 수집 모듈
"""
import os
import json
import time
import requests
import pandas as pd
from typing import Dict, List, Optional, Union, Any


class LottoDataCollector:
    """동행복권 웹 API를 통해 로또 당첨 데이터를 수집하는 클래스"""
    
    BASE_URL = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={}"
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    CACHE_FILE = os.path.join(DATA_DIR, "lotto_data_cache.json")
    
    def __init__(self, use_cache: bool = True):
        """
        LottoDataCollector 초기화
        
        Args:
            use_cache: 캐시 사용 여부 (기본값: True)
        """
        self.use_cache = use_cache
        # 데이터 디렉토리가 없으면 생성
        os.makedirs(self.DATA_DIR, exist_ok=True)
        self.cached_data = self._load_cache() if use_cache else {}
        
    def _load_cache(self) -> Dict[str, Any]:
        """캐시 파일 로드"""
        if os.path.exists(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print("캐시 파일 로드 실패. 새로운 캐시를 생성합니다.")
                return {}
        return {}
    
    def _save_cache(self) -> None:
        """캐시 파일 저장"""
        try:
            with open(self.CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.cached_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"캐시 파일 저장 실패: {e}")
    
    def _fetch_draw(self, draw_number: int) -> Optional[Dict[str, Any]]:
        """
        특정 회차의 당첨 정보를 가져옴
        
        Args:
            draw_number: 조회할 로또 회차
            
        Returns:
            당첨 정보 딕셔너리 또는 None (오류 시)
        """
        # 캐시에 있으면 캐시에서 반환
        cache_key = str(draw_number)
        if self.use_cache and cache_key in self.cached_data:
            return self.cached_data[cache_key]
        
        # 웹 요청
        url = self.BASE_URL.format(draw_number)
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # HTTP 오류 확인
            data = response.json()
            
            # 유효한 응답인지 확인
            if data.get("returnValue") == "success":
                # 캐시에 저장
                if self.use_cache:
                    self.cached_data[cache_key] = data
                    # 주기적으로 캐시 저장 (매 10회 요청마다)
                    if draw_number % 10 == 0:
                        self._save_cache()
                return data
            else:
                print(f"회차 {draw_number}의 데이터가 유효하지 않습니다.")
                return None
                
        except requests.RequestException as e:
            print(f"회차 {draw_number} 데이터 요청 중 오류 발생: {e}")
            return None
        except json.JSONDecodeError:
            print(f"회차 {draw_number} 응답이 유효한 JSON 형식이 아닙니다.")
            return None
    
    def collect_all_draws(self, start_draw: int = 1, end_draw: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        여러 회차의 당첨 정보를 수집
        
        Args:
            start_draw: 시작 회차 (기본값: 1)
            end_draw: 종료 회차 (기본값: 최신 회차)
            
        Returns:
            당첨 정보 딕셔너리 리스트
        """
        # 최신 회차를 찾기 위한 이진 탐색
        if end_draw is None:
            end_draw = self._find_latest_draw()
            
        results = []
        print(f"{start_draw}회차부터 {end_draw}회차까지 데이터를 수집합니다...")
        
        for draw in range(start_draw, end_draw + 1):
            data = self._fetch_draw(draw)
            if data:
                results.append(data)
            # 서버 부하를 줄이기 위한 딜레이
            time.sleep(0.5)
            
            # 진행 상황 표시
            if draw % 10 == 0:
                print(f"현재 진행 상황: {draw}/{end_draw} 회차 수집 완료")
        
        # 최종 캐시 저장
        if self.use_cache:
            self._save_cache()
            
        print(f"총 {len(results)}개의 회차 데이터를 수집했습니다.")
        return results
    
    def _find_latest_draw(self) -> int:
        """
        이진 탐색으로 최신 회차 찾기
        
        Returns:
            최신 회차 번호
        """
        # 이진 탐색 초기 범위 설정
        left, right = 1, 10000  # 충분히 큰 값으로 설정
        latest_found = 1
        
        while left <= right:
            mid = (left + right) // 2
            data = self._fetch_draw(mid)
            
            if data and data.get("returnValue") == "success":
                latest_found = mid
                left = mid + 1  # 더 큰 회차 탐색
            else:
                right = mid - 1  # 더 작은 회차 탐색
                
            # 요청 간 딜레이
            time.sleep(0.5)
        
        return latest_found
    
    def get_data_as_dataframe(self, start_draw: int = 1, end_draw: Optional[int] = None) -> pd.DataFrame:
        """
        당첨 데이터를 DataFrame 형태로 변환
        
        Args:
            start_draw: 시작 회차 (기본값: 1)
            end_draw: 종료 회차 (기본값: 최신 회차)
            
        Returns:
            로또 당첨 정보 DataFrame
        """
        draws_data = self.collect_all_draws(start_draw, end_draw)
        
        # 데이터 변환을 위한 리스트
        processed_data = []
        
        for draw in draws_data:
            # 당첨번호 추출 및 정렬
            numbers = [
                draw.get("drwtNo1"),
                draw.get("drwtNo2"),
                draw.get("drwtNo3"),
                draw.get("drwtNo4"),
                draw.get("drwtNo5"),
                draw.get("drwtNo6")
            ]
            
            # DataFrame용 행 생성
            row = {
                "draw_no": draw.get("drwNo"),
                "date": draw.get("drwNoDate"),
                "numbers": sorted(numbers),
                "bonus": draw.get("bnusNo"),
                "num1": draw.get("drwtNo1"),
                "num2": draw.get("drwtNo2"),
                "num3": draw.get("drwtNo3"),
                "num4": draw.get("drwtNo4"),
                "num5": draw.get("drwtNo5"),
                "num6": draw.get("drwtNo6"),
                "total_sales": draw.get("totSellamnt"),
                "first_prize_amount": draw.get("firstWinamnt"),
                "first_prize_winners": draw.get("firstPrzwnerCo"),
                "first_accum_amount": draw.get("firstAccumamnt")
            }
            processed_data.append(row)
        
        # DataFrame 생성
        return pd.DataFrame(processed_data)
    
    def save_to_csv(self, filename: str = "lotto_data.csv", start_draw: int = 1, end_draw: Optional[int] = None) -> str:
        """
        수집한 로또 데이터를 CSV 파일로 저장
        
        Args:
            filename: 저장할 파일명 (기본값: 'lotto_data.csv')
            start_draw: 시작 회차 (기본값: 1)
            end_draw: 종료 회차 (기본값: 최신 회차)
            
        Returns:
            저장된 파일 경로
        """
        df = self.get_data_as_dataframe(start_draw, end_draw)
        
        # 파일 저장
        filepath = os.path.join(self.DATA_DIR, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"데이터가 {filepath}에 저장되었습니다.")
        
        return filepath 