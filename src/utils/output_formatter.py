"""
로또 번호 예측 결과 출력 형식 모듈
"""
import os
import json
import time
import re
from typing import List, Dict, Any, Optional, Union, TextIO
import matplotlib.pyplot as plt


class OutputFormatter:
    """로또 번호 예측 결과 형식화 클래스"""
    
    def __init__(self, output_dir: str = "results"):
        """
        OutputFormatter 초기화
        
        Args:
            output_dir: 결과 저장 디렉토리 (기본값: 'results')
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def format_combinations(self, 
                           combinations: List[Dict[str, Any]], 
                           start_draw: int, 
                           end_draw: int,
                           include_timestamp: bool = True) -> str:
        """
        번호 조합 결과 텍스트 형식화
        
        Args:
            combinations: 점수화된 번호 조합 리스트
            start_draw: 분석 시작 회차
            end_draw: 분석 종료 회차
            include_timestamp: 타임스탬프 포함 여부
            
        Returns:
            형식화된 출력 문자열
        """
        output = []
        
        # 타임스탬프 추가
        if include_timestamp:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            output.append(f"생성 시간: {timestamp}")
        
        output.append("========== 로또 번호 예측 결과 ==========")
        output.append(f"[분석 기준] {start_draw}회차 ~ {end_draw}회차 데이터 기반\n")
        
        for i, combo in enumerate(combinations, 1):
            # np.int32 제거하고 숫자만 추출
            numbers = combo['numbers']
            clean_numbers = []
            
            for num in numbers:
                # numpy.int32 객체인 경우
                if hasattr(num, 'item'):
                    clean_numbers.append(int(num.item()))
                else:
                    # 문자열에서 숫자만 추출하는 경우
                    if isinstance(num, str) and "np.int32" in num:
                        digit_match = re.search(r'\d+', num)
                        if digit_match:
                            clean_numbers.append(int(digit_match.group()))
                    else:
                        clean_numbers.append(int(num))
            
            # 신뢰도 값 가공 (매우 작은 값이므로 상대적 신뢰도로 표시)
            confidence = combo.get('confidence', 0)
            if confidence < 0.01:  # 매우 작은 값인 경우 상대적 점수로 변환
                # 0~100 사이의 상대적인 신뢰도 점수로 변환 (예시)
                relative_confidence = min(100, confidence * 1e10) if confidence > 0 else 0
                confidence_display = f"{relative_confidence:.1f}점"
            else:
                confidence_display = f"{confidence:.1f}%"
            
            # 깔끔한 형식으로 출력
            output.append(f"게임 {i}: {clean_numbers} (신뢰도: {confidence_display})")
        
        output.append("\n[참고] 본 예측 결과는 확률적 모델에 기반한 것으로,")
        output.append("       당첨을 보장하지 않습니다.")
        
        return "\n".join(output)
    
    def save_text_result(self, 
                        content: str, 
                        filename: Optional[str] = None) -> str:
        """
        텍스트 결과 파일 저장
        
        Args:
            content: 저장할 텍스트 내용
            filename: 파일명 (기본값: 현재 시간 기반 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            filename = f"lotto_prediction_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"결과가 {filepath}에 저장되었습니다.")
        return filepath
    
    def save_json_result(self, 
                        combinations: List[Dict[str, Any]], 
                        metadata: Dict[str, Any],
                        filename: Optional[str] = None) -> str:
        """
        JSON 형식으로 결과 저장
        
        Args:
            combinations: 점수화된 번호 조합 리스트
            metadata: 메타데이터 (시작/종료 회차, 생성 시간 등)
            filename: 파일명 (기본값: 현재 시간 기반 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            filename = f"lotto_prediction_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 저장할 데이터 구성
        data = {
            "metadata": metadata,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "combinations": combinations
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"JSON 결과가 {filepath}에 저장되었습니다.")
        return filepath
    
    def print_result(self, 
                    combinations: List[Dict[str, Any]], 
                    start_draw: int, 
                    end_draw: int,
                    file: Optional[TextIO] = None) -> None:
        """
        결과를 콘솔에 출력
        
        Args:
            combinations: 점수화된 번호 조합 리스트
            start_draw: 분석 시작 회차
            end_draw: 분석 종료 회차
            file: 출력 파일 객체 (기본값: 표준 출력)
        """
        formatted = self.format_combinations(combinations, start_draw, end_draw)
        print(formatted, file=file)
    
    def save_visualization(self, 
                          figure: plt.Figure,
                          filename: Optional[str] = None,
                          dpi: int = 300) -> str:
        """
        시각화 결과 저장
        
        Args:
            figure: Matplotlib Figure 객체
            filename: 파일명 (기본값: 현재 시간 기반 자동 생성)
            dpi: 이미지 해상도
            
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            filename = f"lotto_visualization_{time.strftime('%Y%m%d_%H%M%S')}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        figure.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(figure)
        
        print(f"시각화 결과가 {filepath}에 저장되었습니다.")
        return filepath 