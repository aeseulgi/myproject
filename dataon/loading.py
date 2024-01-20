import numpy as np
import pandas as pd
import random

from datetime import datetime
from datetime import timedelta
import requests 
import csv
import math

from io import StringIO

from time import sleep
import logging

import urllib3

from statsmodels.tsa.stattools import adfuller


class aws:
    
    code_needed = ["402", "403", "413", "415", "421", "510", "889"]
    colnames_AWS = ["time", "temp", "windDir", "windSpd", "precip"]
    code_dict = {"강동": "402", "광진": "413", "송파": "403", "용산": "415", "성동": "421", "영등포": "510", "현충원": "889"}
    columns = ['STN', 'windDir', 'windSpd', 'precip', 'time']
    
    data_dict = dict()
    
    def __init__(self):
        
        self.aws_data = pd.DataFrame(columns = self.columns)
        
    def update_data(self):
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        self.get_past_data()
        
        tm2 = datetime.now()
        tm1 = self.data_dict[self.code_needed[0]]['time'].iloc[len(self.data_dict[self.code_needed[0]])-1]
        tm1 = datetime.strptime(tm1, '%Y-%m-%d %H:%M:%S')
        tm1 += timedelta(hours = 1)
        
        hourly_times = []
        
        cur = tm1
        while cur <= tm2:
            hourly_times.append(cur)
            cur += timedelta(hours=1)

        for time in hourly_times:
            time = time.strftime("%Y%m%d%H%M")
            print("AWS "+time+" has updated")
            temp_data = self.make_aws(time)
            # print(temp_data[temp_data.iloc[:,1] in ["402", "403", "413", "415", "421", "510", "889"]])
            self.aws_data = pd.concat([self.aws_data, self.data_preprocessing(temp_data)], axis = 0)
        
        if hourly_times: # Additional data has joined
            self.data_merge()
            self.save_data()
    
    @classmethod
    def get_past_data(cls):
        for code in cls.code_needed:
            path = "./dataset/climate_data_"+str(code)+".csv"
            cls.data_dict[code] = pd.read_csv(path, header = 0, names = cls.colnames_AWS)
    
    def make_aws(self, time, num_of_seconds_to_wait = 10):
        url = f"https://apihub.kma.go.kr/api/typ01/url/awsh.php?tm={time}&help=0&authKey=ri4L36zbQcuuC9-s27HL3A"
        
        try:
            response_csv = requests.get(url, verify = False)
            random_num_of_seconds = random.randint(num_of_seconds_to_wait, num_of_seconds_to_wait + 3)
            if response_csv.status_code not in (200, 204, 202):
                if random_num_of_seconds <= 50:
                    random_num_of_seconds = random.randint(num_of_seconds_to_wait, num_of_seconds_to_wait + 3)
                    sleep(random_num_of_seconds)
                    return self.make_aws(time, num_of_seconds_to_wait = num_of_seconds_to_wait + 3)
                else:
                    raise Exception(f'Your request failed with the following error: {response_csv.status_code}')
            else:
                aws_data = pd.read_csv(StringIO(response_csv.text))
                return aws_data
        except Exception as e:
            logging.warning(f'Http request failed with url={url}')
            logging.warning(e)
            raise e
    
    @staticmethod
    def data_preprocessing(df):
        df = pd.DataFrame(df)
        # code_needed = ["402", "403", "413", "415", "421", "510", "889"]
        
        column_name = (df.iloc[0, 0].strip().split())[1:]
        data_split = [x.strip().split() for x in df.iloc[2:, 0]]
        aws_data = pd.DataFrame(data_split, columns = column_name)
        aws_data.drop(aws_data.shape[0]-1, axis=0, inplace=True)
        aws_data.reset_index()

        aws_data['time'] = pd.to_datetime(aws_data['YYMMDDHHMI'], format = '%Y%m%d%H%M%S', errors='coerce')
        aws_data = aws_data.rename(columns={'TA': 'temp', 'WD': 'windDir', 'WS': 'windSpd', 'RN_HR1': 'precip'})
        aws_data = aws_data.astype({'temp' : float, 'windDir' : float, 'windSpd' : float, 'precip': float})
        aws_data = aws_data[aws_data['STN'].isin(aws.code_needed)]
        aws_data = aws_data.reset_index()
        aws_data.drop(['index', 'YYMMDDHHMI', 'RN_DAY', 'HM', 'PA', 'PS'], axis = 1, inplace = True)
        aws_data['time'] = aws_data['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # print(aws_data)
        
        return aws_data
    
    def data_merge(self):
        df = self.aws_data
        for code in self.code_needed:
            df_temp = df[df['STN'] == code]
            df_temp = df_temp.reset_index()
            df_temp.drop(['STN', 'index'], inplace = True, axis = 1)
            df_temp = df_temp[['time', 'temp', 'windDir', 'windSpd', 'precip']]
            self.data_dict[code] = pd.concat([self.data_dict[code], df_temp], axis = 0)
            self.data_dict[code] = self.data_dict[code].reset_index()
            self.data_dict[code].drop(['index'], axis = 1, inplace = True)
            
    def save_data(self):
        for code in self.code_needed:
            self.data_dict[code].to_csv(f"./dataset/climate_data_{code}.csv", index = False)
    
    @classmethod
    def save_EDAdata(cls):
        for code in cls.code_needed:
            cls.data_dict[code].to_csv(f"./dataset_cleaning/climate_data_{code}.csv", index = False)
        print("Successfully saved")   
        
    def wind_dir_triangulation(self, name, theta):
        theta = math.radians(theta)
        code = self.code_dict[name]
        self.data_dict[code]["vertical_dir"] = -(self.data_dict[code]["windDir"]+theta).map(math.radians).map(math.cos)
        self.data_dict[code]["horizon_dir"] = (self.data_dict[code]["windDir"]+theta).map(math.radians).map(math.sin)
        
    def wind_dir_triangulation(self, code, theta):
        theta = math.radians(theta)
        self.data_dict[code]["vertical_dir"] = -(self.data_dict[code]["windDir"]+theta).map(math.radians).map(math.cos)
        self.data_dict[code]["horizon_dir"] = (self.data_dict[code]["windDir"]+theta).map(math.radians).map(math.sin)
                                                                                                            
    def apply_windSpd_windDif(self):
        for aws_code in self.code_needed:
            self.data_dict[aws_code]['vertical_dir'] = self.data_dict[aws_code]['vertical_dir'] * self.data_dict[aws_code]['windSpd']
            self.data_dict[aws_code]['horizon_dir'] = self.data_dict[aws_code]['horizon_dir'] * self.data_dict[aws_code]['windSpd']
            self.data_dict[aws_code].drop(['windDir', 'windSpd'], axis = 1, inplace = True)
          
    def missingInput(self):
        for code in self.code_needed:
            print("\n")
            print("code " + str(code) + " proceeding...")
            i = 0
            while self.data_dict[code][['precip', 'horizon_dir', 'vertical_dir', 'temp', 'windSpd']].isnull().values.any():
                print(i, end = "\r")
                i += 1
                self.data_dict[code]['precip'] = self.data_dict[code]['precip'].rolling(window = 3, min_periods = 1).mean()
                self.data_dict[code]['horizon_dir'] = self.data_dict[code]['horizon_dir'].rolling(window = 3, min_periods = 1).mean()
                self.data_dict[code]['vertical_dir'] = self.data_dict[code]['vertical_dir'].rolling(window = 3, min_periods = 1).mean()
                self.data_dict[code]['temp'] = self.data_dict[code]['temp'].interpolate().values
                self.data_dict[code]['windSpd'] = self.data_dict[code]['windSpd'].interpolate().values
                
    @staticmethod         
    def adf_subtest(df, code, col):
        dftest = adfuller(df, autolag="AIC")
        dfoutput = pd.DataFrame({
            "code": code,
            "component": col,
            "Test Statistic": dftest[0],
            "p-value": dftest[1],
            "#Lags Used": dftest[2],
            "Number of Observations Used": dftest[3]
        }, index = [0])

        return dfoutput

    def adf_test(self):
        adf_dict = pd.DataFrame()
        for code in self.code_needed:
            print(str(code) + ": ")
            for i, col in enumerate(self.data_dict[code].columns):
                if i == 0: continue
                print(str(col) + "                    ", end = "\r")
                adf_dict = pd.concat([adf_dict, self.adf_subtest(self.data_dict[code][col], code, col)])
            print("")    
        adf_dict.reset_index()
        return adf_dict
        # new version of aws class
        
        

class lvl():
    code_needed = ["1018640", "1018662", "1018680", "1018683"]
    angle_dict = {"1018640" : 160, "1018662" : 70, "1018680" : 110, "1018683" : 62}
    matching_dict = {"1018640" : ["402", "403", "413"], "1018662": ["403", "413", "421"], "1018680": ["415", "510", "889"], "1018683": ["415", "510", "889"]}
    colnames_lvl = ["lvl", "time"]
    data_dict = dict()
    
    def __init__(self):
        
        return
        
    def update_data(self):
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.get_past_data()
        
        tm2 = datetime.now()
        # tm2 -= timedelta(days = 1)
        # tm2 = tm2.replace(hour = 0, minute = 0, second = 0)
        tm1 = self.data_dict[self.code_needed[0]]['time'].iloc[len(self.data_dict[self.code_needed[0]])-1]
        print(tm1)
        tm1 = tm1[:10]
        tm1 = datetime.strptime(tm1, '%Y-%m-%d')
        tm1 += timedelta(hours = 1)
        
        hour_list = self.generate_date_ranges(tm1, tm2)
    
        for code in self.code_needed:
            for time_list in hour_list:
                print(time_list)
                if type(time_list[0]) != str:
                    time_list_sub = [0, 0]
                    time_list_sub[0] = time_list[0]
                    time_list_sub[1] = time_list[1]
                    time_list_sub[0] = time_list_sub[0].strftime("%Y%m%d")
                    time_list_sub[1] = time_list_sub[1].strftime("%Y%m%d")
                    print(time_list)
                    print("Water level "+code+" "+time_list_sub[0]+"~"+time_list_sub[1]+" has updated")
                    self.data_dict[code] = pd.concat([self.data_dict[code], self.data_preprocessing(self.make_lvl(code, time_list_sub[0], time_list_sub[1]))], axis = 0).reset_index()[:-1]
        
        
        
        if tm1 <= tm2: # Additional data joined.
            self.save_data()
    
    @classmethod
    def get_past_data(cls):
        for code in cls.code_needed:
            path = "./dataset/lvl_data_"+code+".csv"
            try:
                cls.data_dict[code] = pd.read_csv(path, header = 0, names = cls.colnames_lvl) 
            except FileNotFoundError as e:
                print(f'{e}')
                pass 
            
    @staticmethod        
    def generate_date_ranges(startdt, enddt, interval_months=6):
        date_ranges = []
        curdt = startdt

        while curdt < enddt:
            nextdt = min(curdt + timedelta(days=interval_months * 30), enddt)
            date_ranges.append([curdt, nextdt])
            curdt = nextdt
        
        return date_ranges
    
    @staticmethod
    def make_lvl(code, startdt, enddt, num_of_seconds_to_wait = 3):
        url = f"http://www.wamis.go.kr:8080/wamis/openapi/wkw/wl_hrdata"
        
        params = {
            'obscd': f'{code}',
            'startdt': f'{startdt}',
            'enddt': f'{enddt}'
        }
        num_of_seconds_to_wait=3
        random_num_of_seconds = random.randint(num_of_seconds_to_wait, num_of_seconds_to_wait + 3)
        
        try:
            response_csv = requests.get(url, params = params, verify = False)

            if response_csv.status_code not in (200, 204, 202):
                if random_num_of_seconds <= 50: # 50초가 넘어가면 멈춘다.
                    random_num_of_seconds = random.randint(num_of_seconds_to_wait, num_of_seconds_to_wait + 3)
                    sleep(random_num_of_seconds)
                    return make_lvl(code, startdt, enddt,
                                        num_of_seconds_to_wait=num_of_seconds_to_wait + 3)
                else:
                    raise Exception(f'Your request failed with the following error: {response_csv.status_code}')
            else:
                lvl_data = pd.read_csv(StringIO(response_csv.text))
                lvl_data = lvl_data.columns.tolist()
                return lvl_data
        except Exception as e:
            logging.warning(f'Http request failed with url={url}')
            logging.warning(e)
            raise e
        
       
    @staticmethod
    def data_preprocessing(wat_data):
        filtered_wat_data = pd.DataFrame([(wat_data[i], wat_data[i+1]) for i in range(len(wat_data)-1) if i % 2 == 1])
        filtered_wat_data.drop([0], axis = 0, inplace = True)
        filtered_wat_data = filtered_wat_data.rename(columns={0: 'ymdh', 1: 'lvl'})
        filtered_wat_data['ymdh'] = filtered_wat_data['ymdh'].str.extract(r'"ymdh":"(\w{10})"')
        filtered_wat_data['lvl'] = filtered_wat_data['lvl'].str.extract(r'wl:"(.*)"')

        filtered_wat_data['date'] = filtered_wat_data['ymdh'].str[:-2]
        filtered_wat_data['date'] = pd.to_datetime(filtered_wat_data['date'], format='%Y%m%d')
        filtered_wat_data['date'].loc[filtered_wat_data['ymdh'].str.endswith('24')] += timedelta(days = 1)
        filtered_wat_data['ymdh'].loc[filtered_wat_data['ymdh'].str.endswith('24')] = filtered_wat_data['date'].dt.strftime('%Y%m%d')+"00"

        filtered_wat_data['time'] = pd.to_datetime(filtered_wat_data['ymdh'], format = '%Y%m%d%H', errors='coerce')
        filtered_wat_data.reset_index(inplace = True)
        filtered_wat_data.drop(['index', 'ymdh', 'date'], axis = 1, inplace = True)
        filtered_wat_data = filtered_wat_data.astype({'lvl' : float})
        
        
        return filtered_wat_data
    
    def save_data(self):
        for code in self.code_needed:
            self.data_dict[code].to_csv(f"./dataset/lvl_data_{code}.csv", index = False)
            
    @classmethod
    def save_EDAdata(cls):
        for code in cls.code_needed:
            cls.data_dict[code].to_csv(f"./dataset_cleaning/lvl_data_{code}.csv", index = False)
        print("Successfully saved")
            
    @staticmethod         
    def adf_subtest(df, code, col):
        dftest = adfuller(df, autolag="AIC")
        dfoutput = pd.DataFrame({
            "code": code,
            "component": col,
            "Test Statistic": dftest[0],
            "p-value": dftest[1],
            "#Lags Used": dftest[2],
            "Number of Observations Used": dftest[3]
        }, index = [0])

        return dfoutput


    def adf_test(self):
        adf_dict = pd.DataFrame()
        for code in self.code_needed:
            print(str(code) + ": waterlevel")
            adf_dict = pd.concat([adf_dict, self.adf_subtest(self.data_dict[code]["lvl"], code, "lvl")])
            print("")    
        adf_dict.reset_index()
        return adf_dict
    
    def missingInput(self):
        for code in self.code_needed:
            print("\n")
            print("code " + str(code) + " proceeding...")
            i = 0
            while self.data_dict[code][['lvl']].isnull().values.any():
                print(i, end = "\r")
                i += 1
                self.data_dict[code]['lvl'] = self.data_dict[code]['lvl'].interpolate().values
                
