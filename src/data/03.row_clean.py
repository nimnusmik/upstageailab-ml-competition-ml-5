#%%
# import package needed
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re

# set kr font
import matplotlib.font_manager
fontlist = [font.name for font in matplotlib.font_manager.fontManager.ttflist]
# print(fontlist)
krfont = ['Malgun Gothic', 'NanumGothic', 'AppleGothic']
for font in krfont:
    if font in fontlist:
        mpl.rc('font', family=font)


df = pd.read_csv('../../data/processed/cleaned_data/train_clean.csv')

print(df.info())
df_feature_cleaned = df.copy()


#%%
# 교통정보가 없는 행 삭제
df = df.dropna(subset=[
    '지하철최단거리', '버스최단거리',
    '반경_1km_지하철역_수', '반경_1km_버스정류장_수',
    '반경_500m_지하철역_수', '반경_500m_버스정류장_수',
    '반경_300m_지하철역_수', '반경_300m_버스정류장_수'
])

# df.info()



#%%
# 아파트명별 거래 건수 세기
apt_counts = df['아파트명'].value_counts().reset_index()
apt_counts.columns = ['아파트명', '거래건수']

# plt.figure(figsize=(14,6))
# sns.barplot(data=apt_counts, x='아파트명', y='거래건수')

# plt.title('아파트별 거래 건수')
# plt.xlabel('아파트명(거래가 많은 순으로 정렬)')
# plt.ylabel('거래건수')
# plt.xticks([]) 
# plt.tight_layout()
# plt.grid(axis='y')
# plt.show()

# 거래수가 1건인 아파트 제거 (오타, 노이즈 제거)
numtraded_threshold = 1

print(f"-- 해당 아파트에 대한 거래수가 전체 거래수의 {numtraded_threshold}건 이하이면 삭제 --")

rare_apts = apt_counts[apt_counts['거래건수'] <= numtraded_threshold]

rare_apts_list = rare_apts['아파트명'].tolist()
rare_apts_df = df[df['아파트명'].isin(rare_apts_list)]

print(f"Total rows to be deleted: {len(rare_apts_df)}")

display(rare_apts_df[['계약년도','아파트명']])

df = df[~df['아파트명'].isin(rare_apts_list)]
# print(df.info())

########################### 위 코드로 대체 ################################
# # 2023년 이전 드물게 등장한 단지 제거 (오타, 노이즈 제거)
# numtraded_threshold = 0.00001

# print(f"-- 해당 아파트에 대한 거래수가 전체 거래수의 {numtraded_threshold*100}% 미만이면 삭제 --")
# rare_apts = apt_counts[apt_counts['거래건수'] <= apt_counts['거래건수'].sum() * numtraded_threshold]
# rare_apts_list = rare_apts['아파트명'].tolist()
# rare_apts_df = df[(df['아파트명'].isin(rare_apts)) & (df['계약년도'] < 2023)]

# print(f"Total rows to be deleted: {len(rare_apts_df)}")
# display(rare_apts_df)

# df = df[~df['아파트명'].isin(rare_apts_list)]
# display(df.info())
#########################################################################



#%%
# 아파트+전용면적 별 이상치거래 확인

# 이를 위해 아파트가 결측치인 행 먼저 삭제
df = df[df['아파트명'].notna()].copy()

## 전용면적은 오타 방지를 위해 소숫점이하는 버리고 진행

# 판단기준:
## 같은 아파트 및 전용면적별로 각 거래일자의 거래와 앞뒤의 거래를 포함한 11개 target을 이용하여
## robust Z-score를 계산하였을 때 4.5 이상이면 이상치 거래로 판단함

from tqdm import tqdm

threshold = 5
window = 11
result_list = []

df['전용면적_floor'] = np.floor(df['전용면적'])

def robust_z_score_window(x):
    center_value = x[len(x) // 2]
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    if mad == 0:
        return 0
    return 0.6745 * (center_value - median) / mad

for apt, group in tqdm(df.groupby(['아파트명','전용면적_floor'])):
    group = group.sort_values('계약일자').copy()

    group['rolling_robust_z'] = (
        group['target']
        .rolling(window=window, center=True)
        .apply(robust_z_score_window, raw=True)
    )

    group['is_outlier_robust_target'] = group['rolling_robust_z'].abs() > threshold
    result_list.append(group)

outlier_labeled_df = pd.concat(result_list).sort_index()

print(f"이상치 개수: ", len(outlier_labeled_df[outlier_labeled_df['is_outlier_robust_target'] == True]))
display(outlier_labeled_df[outlier_labeled_df['is_outlier_robust_target'] == True])

df = outlier_labeled_df[~outlier_labeled_df['is_outlier_robust_target']]

# print(df.info())



#%%
# 모델링에 필요없는 column제거 후 저장
del_col_list = ['본번', '부번', '번지', '아파트명', 
                '전용면적_floor', 'rolling_robust_z', 'is_outlier_robust_target',
                'isTest']
df = df.drop(columns=del_col_list, errors='ignore')

# display(df.info())
# display(df.head())

df.to_csv('../../data/processed/cleaned_data/train_row_cleaned.csv', index=False, encoding='utf-8')




#######################################################



#%% 
# test_clean.csv에서 교통변수 관련 변수 결측치 보간
test_df = pd.read_csv('../../data/processed/cleaned_data/test_clean.csv')

test_df.info()
display(test_df[test_df['지하철최단거리'].isna()])


#%%
# 전체 데이터에서 교통변수 관련 변수가 결측치 아닌 데이터만 추출
total_df = pd.read_csv('../../data/processed/cleaned_data/total_clean.csv', dtype = {8: 'str',18: 'str',19: 'str',20: 'str',21: 'str'})
transport_cols = [
    '지하철최단거리',
    '반경_1km_지하철역_수',
    '반경_500m_지하철역_수',
    '반경_300m_지하철역_수',
    '버스최단거리',
    '반경_1km_버스정류장_수',
    '반경_500m_버스정류장_수',
    '반경_300m_버스정류장_수'
]
trans_nonnull_df = total_df.dropna(subset=transport_cols)

#%%
# test_df 에서 교통변수 관련변수가 결측치인 데이터의 본번, 부번, 번지들 확인
missing_test_rows = test_df[test_df['지하철최단거리'].isna()][['본번','부번','번지']]
display(missing_test_rows)

#       본번	부번	번지
# 0	    752.0	17.0	752-17
# 1	    780.0	86.0	780-86
# 2	    323.0	4.0	    323-4
# 6	    747.0	34.0	747-34
# 9	    432.0	904.0	432-904
# 3454	38.0	58.0	38-58
# 3471	976.0	15.0	976-15


#%%
# 결측치를 본번이 같은 다른 관측값들의 평균으로 채움
for idx, row in missing_test_rows.iterrows():
    di = row['본번']

    case = trans_nonnull_df[trans_nonnull_df['본번'] == di][transport_cols]
    
    if not case.empty:
        mean_vals = case.mean(numeric_only=True)
        for col in transport_cols:
            test_df.loc[idx, col] = mean_vals[col]

# test_df.info()


# %%
# 모델링에 필요없는 column제거 후 저장
del_col_list = ['본번', '부번', '번지', '아파트명', 
                'isTest']
test_df = test_df.drop(columns=del_col_list, errors='ignore')

# display(test_df.info())
# display(test_df.head())

test_df.to_csv('../../data/processed/cleaned_data/test_na_filled.csv', index=False, encoding='utf-8')

