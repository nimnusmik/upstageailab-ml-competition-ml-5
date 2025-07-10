# --- 1. 라이브러리 설치 및 임포트 ---

# .env 파일을 사용하기 위한 라이브러리를 설치합니다.
#!pip install python-dotenv

import pandas as pd
import requests
import time
from tqdm import tqdm
import os
from dotenv import load_dotenv # .env 파일을 읽기 위한 라이브러리
from datetime import datetime
import re

# tqdm 라이브러리와 pandas의 연동을 설정합니다.
tqdm.pandas()

# --- 2. 오류 메시지 출력 함수 & 로그(출력문) 저장 함수 ---

# 오류 메시지를 빨간색으로 출력하는 함수 정의
def print_error(text):
    # ANSI 이스케이프 코드: '\033[91m'는 빨간색 출력, '\033[0m'는 색상 초기화
    print(f"\033[91m{text}\033[0m")


# 로그 파일이 저장될 디렉토리 경로 설정
LOG_DIR = "../data/logs"

# 로그 디렉토리가 존재하지 않으면 생성
os.makedirs(LOG_DIR, exist_ok=True)

# 로그 파일명: 실행 시간 기준으로 생성 (예: geocoding_run_20250710_112530.log)
log_filename = f"train_coordinate_geocoding_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 전체 로그 파일 경로 구성
LOG_PATH = os.path.join(LOG_DIR, log_filename)


# 로그를 파일로 저장하고, 필요 시 콘솔에도 출력하는 함수
def write_log(message, log_path=LOG_PATH, print_also=True):
    # 로그 파일을 append 모드로 열고 메시지를 시간과 함께 기록
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {message}\n")
    
    # 옵션에 따라 콘솔에도 출력 (기본값: True)
    if print_also:
        print(message)
        

# --- 3. .env 파일에서 API 키 로드 ---

# .env 파일에 정의된 환경 변수를 현재 세션으로 로드합니다.
load_dotenv()

# API 키 관리를 위한 전역 변수
key_idx = 0
all_keys_exhausted = False

# KAKAO_REST_API_KEY1부터 KAKAO_REST_API_KEY10까지의 키를 저장할 리스트
KAKAO_KEYS = []

# 여러 개의 API 키를 리스트로 불러오기
for i in range(1, 11):  # 최대 10개 지원 (필요시 숫자 변경)
    # os.getenv() 함수를 사용하여 "KAKAO_REST_API_KEY" 라는 이름의 환경 변수 값을 가져옵니다.
    key = os.getenv(f"KAKAO_REST_API_KEY{i}")
    if key: KAKAO_KEYS.append(key)

if not KAKAO_KEYS:
    print_error("[오류] '.env 파일에 KAKAO_REST_API_KEY1~N 형식으로 키가 필요합니다.")
    write_log("[오류] '.env 파일에 KAKAO_REST_API_KEY1~N 형식으로 키가 필요합니다.",print_also=False)
else:
    write_log(f"총 {len(KAKAO_KEYS)}개의 API 키 로드 완료.")


# --- 4. 경로 설정 및 데이터 로드후 좌표 결측치 개수 확인 ---

# 원본 학습 데이터 경로 지정
TRAIN_DATA_PATH = '../data/raw/train.csv'

# 처리된 데이터가 저장될 디렉토리 경로
OUTPUT_DIR = '../data/processed/'

# 도로명 주소가 없는 행을 저장할 파일명
NO_ADDRESS_FILENAME = 'missing_address_rows.csv'

NO_SEARCH_FILENAME = 'no_search_geocording.csv'

# 중간 저장 체크포인트 파일명
CHECKPOINT_FILENAME = 'train_geocoded_checkpoint.csv'

# 최종 결과 파일명
FINAL_FILENAME = 'train_geocoded.csv'

# 각 파일의 전체 경로 구성
NO_ADDRESS_PATH = os.path.join(OUTPUT_DIR, NO_ADDRESS_FILENAME)
NO_SEARCH_PATH = os.path.join(OUTPUT_DIR, NO_SEARCH_FILENAME)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, CHECKPOINT_FILENAME)
FINAL_PATH = os.path.join(OUTPUT_DIR, FINAL_FILENAME)

# --- 카카오 API URL 상수 정의 ---
KAKAO_GEOCODE_URL = "https://dapi.kakao.com/v2/local/search/address.json"

# 데이터 로드 시작 메시지 출력
write_log(f"'{TRAIN_DATA_PATH}'에서 원본 데이터를 로드합니다...")

try:
    # 원본 CSV 파일을 pandas로 읽어옴
    train_df = pd.read_csv(TRAIN_DATA_PATH, low_memory=False)
    write_log("데이터 로드 완료.")
except FileNotFoundError:
    # 파일이 존재하지 않을 경우 예외 처리
    print_error(f"오류: 'except FileNotFoundError:' / '{TRAIN_DATA_PATH}' 파일을 찾을 수 없습니다.")
    write_log(f"오류: '{TRAIN_DATA_PATH}' 파일을 찾을 수 없습니다.", print_also=False)
    train_df = None
    
# 좌표X, 좌표Y 결측치 개수 출력 (isnull을 먼저 사용하고 isnull로 조회되지 않는 결측치 모두 조회)
if train_df is not None:
    missing_x_count = train_df['좌표X'].isnull().sum()
    missing_y_count = train_df['좌표Y'].isnull().sum()
    write_log(f"좌표X 결측치 개수: {missing_x_count}")
    write_log(f"좌표Y 결측치 개수: {missing_y_count}")

    # 좌표X, 좌표Y 모두 결측치인 행 개수 출력
    missing_both_count = train_df[(train_df['좌표X'].isnull()) & (train_df['좌표Y'].isnull())].shape[0]
    write_log(f"좌표X와 좌표Y 모두 결측치인 행 개수: {missing_both_count}")


# --- 5. 주소 생성 함수 정의 (도로명/번지 예외 처리) ---

def create_full_address(row):
    """
    도로명 주소를 우선 사용하되, 없으면 번지 주소를 사용합니다.
    """
    # '도로명'이 유효한 문자열인지 확인
    if isinstance(row['도로명'], str) and row['도로명'].strip():
        return row['시군구'] + ' ' + row['도로명']
    # '도로명'이 없으면 '번지'가 유효한지 확인
    elif isinstance(row['번지'], str) and row['번지'].strip():
        return row['시군구'] + ' ' + row['번지']
    # 둘 다 없으면 None 반환
    else:
        return None
    
    
def remove_dong(address):
    """
    주소 문자열에서 '동/가/로'로 끝나는 세 번째 부분을 제거합니다.
    이 행정구역 명칭이 주소 검색을 방해하는 경우가 있기 때문입니다.
    예: '서울특별시 강남구 개포동 언주로 21' → '서울특별시 강남구 언주로 21'
    """
    try:
        parts = address.split()
        # 도로명 주소의 'OO길'을 제거하지 않도록 '동', '가', '로'만 대상으로 한정합니다.
        if len(parts) > 2 and (parts[2].endswith(('동', '가', '로'))):
            return ' '.join(parts[:2] + parts[3:])
    except Exception as e:
        write_log(f"[오류] 주소 변형 중 오류 발생: {e}")
        
     # 변형에 실패하거나 조건에 맞지 않으면 원본 주소 반환
    return address




# --- 6. 지오코딩 및 주소 생성 함수 정의 ---

def get_coordinates(address, key_index):
    """
    (단순 작업자) 카카오 API를 활용하여 '하나의' 주어진 주소를 좌표로 변환합니다.
    """
    api_key = KAKAO_KEYS[key_index]
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    params = {"query": address}
    headers = {"Authorization": f"KakaoAK {api_key}"}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if data['documents']:
            return float(data['documents'][0]['x']), float(data['documents'][0]['y'])
        else:
            return None # API가 주소를 못 찾으면 None 반환
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code in [403, 429]: return "CHANGE_KEY"
        write_log(f"[실패] API 요청 실패 (HTTP 오류): {e}, 주소: '{address}'")
        return None
        
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        write_log(f"[네트워크 오류] 연결 실패. 주소: '{address}'")
        time.sleep(5)
        return "RETRY"
        
    except Exception as e:
        write_log(f"[실패] 알 수 없는 오류: {e}, 주소: '{address}'")
        return None  

def geocode_with_key_rotation(address):
    """
    (현장 관리자) API 키를 순환하며 지오코딩을 시도하고, 실패 시 주소를 변형해 재시도합니다.
    """
    global key_idx, all_keys_exhausted
    if pd.isnull(address):
        return None

    # 1. 시도할 주소 목록 생성 (원본, 단축 주소)
    addresses_to_try = [address]
    short_address = remove_dong(address)
    if short_address != address:
        addresses_to_try.append(short_address)

    # 2. for 문으로 주소 목록을 순회 (여기서 'i' 변수가 올바르게 사용됩니다)
    for i, current_address in enumerate(addresses_to_try):
        if i > 0: # 2차 시도일 경우 로그 출력
            write_log(f"  [재시도] 주소 변형 후 검색: '{current_address}'")

        start_key_idx = key_idx
        retry_attempts = 0
        
        # 3. while 문으로 현재 주소에 대한 키 교체 및 네트워크 재시도 처리
        while True:
            if all_keys_exhausted: return None
            
            result = get_coordinates(current_address, key_idx)

            if result == "CHANGE_KEY":
                key_idx = (key_idx + 1) % len(KAKAO_KEYS)
                if key_idx == start_key_idx:
                    all_keys_exhausted = True; break
                time.sleep(1); continue
            
            elif result == "RETRY":
                retry_attempts += 1
                if retry_attempts >= 5: break
                time.sleep(5); continue
            
            # 4. 최종 결과 처리
            if result: # 좌표를 성공적으로 찾은 경우
                if i > 0: # 2차 시도에서 성공했다면 로그 남기기
                    write_log(f"  [2차 성공] '{current_address}' 검색으로 좌표를 찾았습니다.")
                return result
            else: # API가 주소를 못 찾은 경우(None)
                write_log(f"  [검색 실패] API가 '{current_address}' 주소의 결과를 찾지 못했습니다.")
                break # 내부 while 루프 탈출 후 다음 주소 시도
    
    return None # 모든 주소 시도가 실패한 경우



# --- 7. 결측치 채우기 실행 ---

# 데이터프레임이 정상적으로 로드되었고, API 키가 존재하는 경우에만 실행
if train_df is not None and KAKAO_KEYS:
    
    # 좌표가 비어있는 행만 복사하여 작업 대상으로 지정
    df_to_geocode = train_df[train_df['좌표X'].isnull()].copy()
    
    if not df_to_geocode.empty:
        write_log(f"\n총 {len(df_to_geocode)}개의 좌표 결측치에 대한 작업을 시작합니다...")

        # 7.1. 주소 생성 및 주소 없는 행 분리/저장
        # 도로명 또는 번지를 기준으로 전체 주소를 생성하고
        # full_address가 없는 행은 별도로 저장
        df_to_geocode['full_address'] = df_to_geocode.apply(create_full_address, axis=1)
        no_address_rows = df_to_geocode[df_to_geocode['full_address'].isnull()]

        if not no_address_rows.empty:
            # 저장 디렉토리가 없으면 생성
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            # 도로명/번지 모두 없는 행 저장
            no_address_rows.to_csv(NO_ADDRESS_PATH, index=False)
            write_log(f"{len(no_address_rows)}개의 주소 없는 행을 '{NO_ADDRESS_PATH}'에 저장했습니다.")
        else:
            write_log("모든 행에 '도로명' 또는 '번지' 주소 정보가 존재합니다.")

        # 주소가 존재하는 행만 남기고 필터링
        df_to_geocode.dropna(subset=['full_address'], inplace=True)

        # 7.2. 고유 주소에 대한 API 호출 (캐싱)
        # 중복된 주소는 API 호출을 1회만 하기 위해 고유한 주소만 추출
        unique_addresses = df_to_geocode['full_address'].unique()
        write_log(f"실제 API 호출 대상 고유 주소 개수: {len(unique_addresses)}")
        
        # 주소별 좌표를 저장할 캐시 딕셔너리
        address_cache = {}
        processed_count = 0  # 처리된 주소 수 카운트

        # 고유 주소 목록을 순회하며 지오코딩 수행
        for addr in tqdm(unique_addresses, desc="Geocoding Progress"):
            # 모든 키가 소진되었으면 루프 중단
            if all_keys_exhausted:
                print_error("\n!! 모든 키 사용량 소진: 작업 중단 !!")
                write_log("\n모든 키 사용량 소진: 작업 중단", print_also=False)
                break
            
            # 현재 주소에 대해 좌표 요청
            coords = geocode_with_key_rotation(addr)

            # 좌표를 성공적으로 받아온 경우 캐시에 저장
            if coords:
                address_cache[addr] = coords
                processed_count += 1
            
            # 7.3. 1000개 단위로 임시 저장 (체크포인트)
            if processed_count > 0 and processed_count % 1000 == 0:
                write_log(f"\n--- {processed_count}개 처리 완료. 체크포인트를 저장합니다 ---")
                
                # 원본 데이터 복사 후 full_address 다시 생성
                temp_df = train_df.copy()
                temp_df['full_address_map'] = temp_df.apply(create_full_address, axis=1)
                
                # 캐시된 주소를 기반으로 좌표 매핑
                temp_coords = temp_df['full_address_map'].map(address_cache)
                valid_coords_map = temp_coords.dropna()

                # 좌표 값 반영
                temp_df.loc[valid_coords_map.index, '좌표X'] = [x[0] for x in valid_coords_map]
                temp_df.loc[valid_coords_map.index, '좌표Y'] = [x[1] for x in valid_coords_map]

                # 임시 컬럼 제거 후 저장
                temp_df.drop(columns=['full_address_map'], inplace=True)
                temp_df.to_csv(CHECKPOINT_PATH, index=False)
                write_log(f"체크포인트 저장 완료: '{CHECKPOINT_PATH}'")

        # 7.4. 캐싱된 결과 매핑 및 최종 업데이트
        write_log("\n최종 결과 업데이트를 시작합니다...")

        # df_to_geocode의 주소를 기준으로 캐시된 좌표 매핑
        final_coords_map = df_to_geocode['full_address'].map(address_cache)
        valid_final_coords = final_coords_map.dropna()

        # 원본 train_df의 해당 인덱스에 좌표 반영
        train_df.loc[valid_final_coords.index, '좌표X'] = [x[0] for x in valid_final_coords]
        train_df.loc[valid_final_coords.index, '좌표Y'] = [x[1] for x in valid_final_coords]
        write_log("결측치 업데이트 완료!")
        
        # 7.5. 최종 검색 실패 행 식별
        cached_addresses = set(address_cache.keys())
        all_geocoding_addresses = set(unique_addresses)
        failed_addresses = all_geocoding_addresses - cached_addresses
        
        if failed_addresses:
            # df_to_geocode에서 최종 실패한 행들을 failed_rows 변수에 저장
            failed_rows = df_to_geocode[df_to_geocode['full_address'].isin(failed_addresses)].copy()
            write_log(f"총 {len(failed_rows)}개 주소의 좌표 검색에 최종 실패했습니다.")

    else:
        write_log("\n좌표에 결측치가 없어 추가 작업을 진행하지 않습니다.")
        
        
        
# --- 8. 결과 확인 및 저장 ---

# train_df가 정상적으로 존재할 경우에만 저장 진행
if train_df is not None:
    if not failed_rows.empty:
        # 1. 실패 행을 별도 CSV 파일로 저장
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        failed_rows.to_csv(NO_SEARCH_PATH, index=False)
        write_log(f"검색 실패 행 {len(failed_rows)}개를 '{NO_SEARCH_PATH}'에 저장했습니다.")
        
        # 2. 원본 데이터프레임에서 실패 행 삭제
        train_df.drop(index=failed_rows.index, inplace=True)
        write_log(f"최종 데이터에서 검색 실패 행 {len(failed_rows)}개를 삭제했습니다.")
    
    # 좌표X의 남은 결측치 개수를 최종 확인
    final_missing_count = train_df['좌표X'].isnull().sum()
    write_log(f"\n작업 후 남은 '좌표X'의 결측치 개수: {final_missing_count}")

    # 저장 디렉토리가 없을 경우 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 전체 train_df를 최종 결과 파일로 저장
    train_df.to_csv(FINAL_PATH, index=False)
    write_log(f"모든 작업이 완료되었습니다. 최종 데이터가 '{FINAL_PATH}' 경로에 저장되었습니다.")

else:
    # train_df가 None이면 에러 메시지 출력 후 저장하지 않음
    print_error("\n[작업 중단] 데이터 파일이 없어 최종 저장을 진행할 수 없습니다.")
    write_log("[작업 중단] 데이터 파일이 없어 최종 저장을 진행할 수 없습니다.", print_also=False)