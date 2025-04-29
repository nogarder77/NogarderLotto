"""
로또 번호 예측 프로그램 메인 모듈
"""
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 상대 경로로 임포트
from data.collector import LottoDataCollector
from data.preprocessor import DataPreprocessor
from model.cdm_model import CDMModel
from model.number_generator import NumberGenerator
from utils.output_formatter import OutputFormatter
from utils.helpers import save_combination_history


def parse_arguments() -> argparse.Namespace:
    """
    명령줄 인자 파싱
    
    Returns:
        파싱된 인자
    """
    parser = argparse.ArgumentParser(description='로또 번호 예측 프로그램')
    
    parser.add_argument('-s', '--start-draw', type=int, default=1,
                        help='데이터 수집 시작 회차 (기본값: 1)')
    
    parser.add_argument('-e', '--end-draw', type=int, default=None,
                        help='데이터 수집 종료 회차 (기본값: 최신)')
    
    parser.add_argument('-n', '--num-combinations', type=int, default=5,
                        help='생성할 번호 조합 개수 (기본값: 5)')
    
    parser.add_argument('-c', '--use-cache', action='store_true',
                        help='캐시 사용 여부')
    
    parser.add_argument('-o', '--output-file', type=str, default=None,
                        help='결과 저장 파일명 (기본값: 자동 생성)')
    
    parser.add_argument('-f', '--format', type=str, choices=['text', 'json', 'both'], default='text',
                        help='결과 저장 형식 (text/json/both, 기본값: text)')
    
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='번호 확률 시각화 저장 여부')
    
    parser.add_argument('-d', '--diverse', action='store_true',
                        help='다양성 있는 번호 조합 생성 여부')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='난수 생성 시드')
    
    return parser.parse_args()


def run_prediction(args: argparse.Namespace) -> None:
    """
    예측 실행
    
    Args:
        args: 명령줄 인자
    """
    print("========== 로또 번호 예측 프로그램 ==========")
    start_time = time.time()
    
    # 1. 데이터 수집
    print("\n[1/4] 로또 당첨 데이터 수집 중...")
    collector = LottoDataCollector(use_cache=args.use_cache)
    
    if args.end_draw is None:
        # 최신 회차 찾기
        latest_draw = collector._find_latest_draw()
        print(f"최신 회차는 {latest_draw}회차입니다.")
        args.end_draw = latest_draw
    
    # CSV 파일로 저장
    csv_file = collector.save_to_csv(start_draw=args.start_draw, end_draw=args.end_draw)
    
    # 2. 데이터 전처리
    print("\n[2/4] 데이터 전처리 중...")
    preprocessor = DataPreprocessor(csv_file)
    
    # 번호 조합 추출
    combinations = preprocessor.get_number_combinations()
    print(f"총 {len(combinations)}개의 당첨 번호 조합을 추출했습니다.")
    
    # 데이터 패턴 분석
    analysis = preprocessor.analyze_number_patterns()
    print("당첨 번호 패턴 분석 완료")
    
    # 3. CDM 모델링
    print("\n[3/4] CDM 모델 학습 중...")
    # 사전 확률 설정 (균등 분포)
    prior_alpha = np.ones(45)  # 1~45까지 균등 확률
    
    # CDM 모델 초기화 및 학습
    cdm_model = CDMModel(prior_alpha=prior_alpha, seed=args.seed)
    cdm_model.fit(combinations)
    
    # 번호별 확률 확인
    probabilities = cdm_model.get_number_probabilities()
    
    # 시각화 저장 (선택적)
    if args.visualize:
        print("번호별 확률 시각화 생성 중...")
        figure = cdm_model.plot_number_probabilities()
        output_formatter = OutputFormatter()
        output_formatter.save_visualization(figure)
    
    # 4. 번호 생성
    print("\n[4/4] 번호 조합 생성 중...")
    generator = NumberGenerator(cdm_model=cdm_model, seed=args.seed)
    
    # 다양성 사용 여부에 따라 다른 방식으로 조합 생성
    if args.diverse:
        print("다양성 있는 번호 조합 생성 중...")
        # 초기 조합 수를 더 많이 생성
        initial_combinations = cdm_model.generate_multiple_combinations(
            n_combinations=args.num_combinations * 3
        )
        
        # 점수화
        all_scored = generator.score_combinations(initial_combinations)
        
        # 유사성이 낮은 조합 선택
        final_combinations = []
        selected_indices = []
        
        for i, combo in enumerate(all_scored):
            # 이미 선택된 조합들과 비교
            numbers = combo['numbers']
            
            # 처음 선택하는 경우 또는 기존 선택과 중복이 적은 경우
            if not final_combinations or not any(
                len(set(numbers).intersection(set(final_combinations[j]['numbers']))) >= 4 
                for j in selected_indices
            ):
                final_combinations.append(combo)
                selected_indices.append(i)
                
                # 충분한 조합을 선택했으면 종료
                if len(final_combinations) >= args.num_combinations:
                    break
    else:
        # 기본 생성 방식
        final_combinations = generator.generate_best_combinations(
            n_to_select=args.num_combinations
        )
    
    # 결과 출력 및 저장
    output_formatter = OutputFormatter()
    
    # 콘솔 출력
    output_formatter.print_result(final_combinations, args.start_draw, args.end_draw)
    
    # 파일 저장
    if args.format in ['text', 'both']:
        output_formatter.save_text_result(
            output_formatter.format_combinations(final_combinations, args.start_draw, args.end_draw),
            args.output_file
        )
    
    if args.format in ['json', 'both']:
        metadata = {
            'start_draw': args.start_draw,
            'end_draw': args.end_draw,
            'model': 'CDM',
            'seed': args.seed
        }
        json_file = args.output_file.replace('.txt', '.json') if args.output_file else None
        output_formatter.save_json_result(final_combinations, metadata, json_file)
    
    # 히스토리 저장
    save_combination_history(final_combinations)
    
    # 실행 시간 출력
    elapsed_time = time.time() - start_time
    print(f"\n총 실행 시간: {elapsed_time:.2f}초")


if __name__ == "__main__":
    try:
        args = parse_arguments()
        run_prediction(args)
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        sys.exit(1) 