# !pip install geopy

import os
import warnings
from datetime import datetime, time
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from geopy.distance import geodesic
import pytz

warnings.filterwarnings('ignore')
tqdm.pandas()


class Preprocessing:
    def __init__(
            self,
            dfs,    # 전체 데이터셋
            df_asn,  # ASN 데이터셋
            df_timezone,    # 국가별 timezone 데이터셋
    ):
        self.df_asn = df_asn
        self.df_timezone = df_timezone

        # 0. column reset
        self.dfs_ = dfs[[
            'timestamp', 'user_id', 'name', 'country', 'region_x', 'city', 'asn', 'latitude', 'longitude',
            'browser_name_and_version', 'os_name_and_version', 'device_type', 'rtt',
            'login_success', 'is_attack_ip', 'is_takeover', 'label'
        ]]
        self.dfs_.rename(
            columns={
                'country': 'country_code', 'name': 'country', 'region_x': 'region',
            }, inplace=True
        )

        # 1. Region: fill nan values and make grade feature considering the attack rate by region
        self.dfs_['region'] = np.where(
            (self.dfs_['region'] == '-') | self.dfs_['region'].isna(),
            'unknown', self.dfs_['region']
        )
        self.dfs_ = self.get_grade('region')

        # 2. city: fill nan values and make grade feature considering the attack rate by city
        self.dfs_['city'] = np.where(
            (self.dfs_['city'] == '-') | self.dfs_['city'].isna(),
            'unknown', self.dfs_['city']
        )
        self.dfs_ = self.get_grade('city')

        # 3. asn: merge asn dataset, then get
        self.df_asn.columns = ['start', 'end', 'asn', 'name']
        df_asn_ = self.df_asn[['asn', 'name']]
        df_asn_ = df_asn_.groupby('asn').max().reset_index(drop=False)
        self.dfs_ = pd.merge(left=self.dfs_, right=df_asn_, how='left', on='asn')
        self.dfs_['name'] = np.where(self.dfs_['name'].isna(), 'unknown', self.dfs_['name'])
        self.dfs_ = self.get_grade('name')

        # 4. browser_name_and_version:
        # Extract only the browser name and Create a feature indicating whether the browser is a legacy browser
        self.dfs_['browser_name'] = self.dfs_['browser_name_and_version'].apply(
            lambda x: x if len(x.split()) == 1 else ' '.join(x.split()[:-1])
        )
        legacy_browsers = ['Chrome', 'Firefox', 'Internet Explorer', 'Safari', 'Android', 'Opera', 'Edge',
                           'Samsung Internet', 'IE']
        self.dfs_['browser_is_legacy'] = self.dfs_['browser_name'].apply(
            lambda x: any(browser in x for browser in legacy_browsers))

        # 5. os_name_and_version:
        # Extract only the os name and Create a feature indicating whether the os is a legacy os
        self.dfs_['os_name'] = self.dfs_['os_name_and_version'].apply(
            lambda x: x if len(x.split()) == 1 else ' '.join(x.split()[:-1])
        )
        legacy_os = ['iOS', 'Android', 'Mac OS X', 'Chrome OS', 'Windows Phone', 'Windows', 'Ubuntu']
        self.dfs_['os_is_legacy'] = self.dfs_['os_name'].apply(lambda x: any(os_name in x for os_name in legacy_os))

        # 6. rtt:
        # First, fill NaN values in 'rtt' with the median 'rtt' of the same country.
        # If there are still NaNs, fill them using the median 'rtt' of the closest country based on lat and lon.
        self.dfs_ = self.fill_rtt_with_asn()

        # 7. device type: fill nan
        self.dfs_['device_type'] = np.where(self.dfs_['device_type'].isna(), 'unknown', self.dfs_['device_type'])

        # 8. timestamp
        # Use international standard time data to create a variable that determines
        # whether it is evening time (9 PM to 5 AM) for each country.
        # self.dfs_ = self.get_is_night()

        # 9. Final column cleanup
        self.dfs_ = self.dfs_[[
            'user_id', 'country', 'country_code', 'region', 'region_risk_grade', 'city', 'city_risk_grade',
            'name', 'name_risk_grade', 'browser_name_and_version', 'browser_name', 'browser_is_legacy',
            'os_name_and_version', 'os_name', 'os_is_legacy', 'device_type', 'rtt', 'timestamp',  # 'is_evening',
            'login_success', 'is_attack_ip', 'is_takeover', 'label',
        ]]
        self.dfs_.rename(columns={'name': 'asn_name', 'name_risk_grade': 'asn_risk_grade'})
        # convert T/F to 1/0
        columns_to_convert = ['browser_is_legacy', 'os_is_legacy',
                              'login_success', 'is_attack_ip', 'is_takeover', 'label']
        self.dfs_ = self.convert_boolean_columns(columns_to_convert)


    def get_grade(self, target_col):
        def assign_risk_category(row):
            if row['mean'] == 1:
                return 5
            elif row['mean'] >= 0.75:
                return 4
            elif row['mean'] >= 0.5:
                return 3
            elif row['mean'] >= 0.25:
                return 2
            elif row['mean'] > 0:
                return 1
            else:
                return 0

        df = self.dfs_.groupby(target_col)['label'].agg(['mean', 'size']).sort_values(ascending=False,
                                                                                      by='mean').reset_index()
        col_nm = target_col + '_risk_grade'
        df[col_nm] = df.apply(assign_risk_category, axis=1)
        df_ = self.dfs_.merge(df[[target_col, col_nm]], on=target_col, how='left')
        return df_

    def fill_rtt_with_asn(self):
        def fill_rtt(row, valid_coor):
            """
            거리 계산 및 NaN 채우기 함수 정의
            :param row:
            :param valid_coor:
            :return:
            """
            if pd.isna(row['rtt']):
                # 거리 계산
                distances = valid_coor.apply(
                    lambda x: geodesic((x['latitude'], x['longitude']), (row['latitude'], row['longitude'])).kilometers,
                    axis=1
                )
                # 가장 가까운 국가의 'rtt' 중간값으로 채우기
                closest_idx = distances.idxmin()
                return valid_coor.loc[closest_idx, 'rtt']
            else:
                return row['rtt']
        df = self.dfs_
        df['asn'] = df.groupby('country')['rtt'].transform(lambda x: x.fillna(x.median()))

        df_a = df.groupby(['country', 'longitude', 'latitude'])['rtt'].median().reset_index()

        # 'rtt' 값이 있는 국가들의 중심 좌표만 추출
        valid_coords = df_a.dropna(subset=['rtt'])
        df_a['rtt'] = df_a.apply(lambda row: fill_rtt(row, valid_coords), axis=1)
        df_ = pd.merge(df, df_a[['country', 'rtt']], on='country', how='left')
        df_.drop(columns=['rtt_x'], inplace=True)
        df_.rename(columns={'rtt_y': 'rtt'}, inplace=True)
        return df_

    def convert_boolean_columns(self, columns_to_convert):
        df = self.dfs_
        for column in columns_to_convert:
            # True/False 값을 1/0으로 변환
            df[column] = df[column].astype(int)
        return df

    # def get_is_night(self):
    #
    #     def is_evening(timestamp, country_code):
    #         if timestamp.tzinfo is None:
    #             timestamp = timestamp.tz_localize('UTC')
    #
    #         # 국가 코드에 해당하는 시간대 찾기
    #         timezone_str = country_to_timezone.get(country_code, 'UTC')  # 기본값으로 UTC 설정
    #         timezone = pytz.timezone(timezone_str)
    #
    #         # 타임스탬프를 해당 국가의 시간대로 변환
    #         local_time = timestamp.astimezone(timezone)
    #
    #         # 저녁 시간 정의 (오후 9시부터 새벽 5시 이전)
    #         if time(23, 0) <= local_time.time() or local_time.time() < time(2, 0):
    #             return True
    #         else:
    #             return False
    #
    #     df = self.dfs_
    #     country_to_timezone = {row['country_code']: row['zone'] for index, row in self.df_timezone.iterrows()}
    #     df['timestamp'] = pd.to_datetime(df['timestamp'])
    #     df['is_evening'] = df.apply(lambda row: is_evening(row['timestamp'], row['country_code']), axis=1)
    #     return df


cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'data')
df_asia = pd.read_csv(os.path.join(data_dir, 'df_asia.csv'))
df_asn_nm = pd.read_csv(os.path.join(data_dir, 'asn-ipv4.csv'), header=None)
df_tz = pd.read_csv(os.path.join(data_dir, 'timezone_countries.csv'))

dfs_preprocessed = Preprocessing(
    dfs=df_asia,
    df_asn=df_asn_nm,
    df_timezone=df_tz).dfs_
dfs_preprocessed.to_csv(os.path.join(data_dir, 'preprocessed.csv'), encoding='utf-8-sig', index=False)