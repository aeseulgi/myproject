import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
from datetime import timedelta
from datetime import datetime

from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.seasonal import DecomposeResult


class dataMSTL:
    
    def __init__(self, type_df, code):
        
        self.type_data = type_df
        self.loc_code = code
        self.raw_data = pd.read_csv(f"./dataset_cleaning/{type_df}_data_{code}.csv")

        self.trend = pd.DataFrame()
        self.se24 = pd.DataFrame()
        self.se360 = pd.DataFrame()
        self.se8760 = pd.DataFrame()
        self.resid = pd.DataFrame()
        self.mstl_data = [
            self.trend,
            self.se24,
            self.se360,
            self.se8760,
            self.resid
        ]
        
        
    def my_mstl(self, periods = [24, 24*15, 24*365]):
        timeseries_df = self.raw_data
        
        for i, col in enumerate(timeseries_df.columns):
            if col in ["time", "index"]: continue
            print(str(self.type_data) + " " + str(self.loc_code) + " " + str(col) + " proceeding...               ", end = "\r")

            mstl = MSTL(timeseries_df[col], periods = periods)
            mstl_data = mstl.fit()
            mstl_data.plot()
            mstl_data_trend = mstl_data.trend
            mstl_data_seasonal = mstl_data.seasonal
            mstl_data_residual = mstl_data.resid
            
            self.trend = pd.concat(
                [self.trend, mstl_data_trend],
                axis = 1
            )
            self.se24 = pd.concat(
                [self.se24, mstl_data_seasonal["seasonal_24"]],
                axis = 1
            )
            self.se360 = pd.concat(
                [self.se360, mstl_data_seasonal["seasonal_360"]],
                axis = 1
            )
            self.se8760 = pd.concat(
                [self.se8760, mstl_data_seasonal["seasonal_8760"]],
                axis = 1
            )
            self.resid = pd.concat(
                [self.resid, mstl_data_residual],
                axis = 1
            )
            
        print("")
    
    def my_mstl_sub(self, periods = [24*15]):
        timeseries_df = self.raw_data
        
        for i, col in enumerate(timeseries_df.columns):
            if col in ["time", "index"]: continue
            print(str(self.type_data) + " " + str(self.loc_code) + " " + str(col) + " proceeding...               ", end = "\r")

            mstl = MSTL(timeseries_df[col], periods = periods)
            mstl_data = mstl.fit()
            mstl_data.plot()
            mstl_data_trend = mstl_data.trend
            mstl_data_seasonal = mstl_data.seasonal
            mstl_data_residual = mstl_data.resid
            
            self.trend = pd.concat(
                [self.trend, mstl_data_trend],
                axis = 1
            )
            self.se24 = pd.concat(
                [self.se24, mstl_data_seasonal],
                axis = 1
            )
            self.se8760 = pd.concat(
                [self.se8760, mstl_data_seasonal["seasonal_8760"]],
                axis = 1
            )
            self.resid = pd.concat(
                [self.resid, mstl_data_residual],
                axis = 1
            )
            
        print("")
        
        
    def my_stl(self, period = 2190):
        timeseries_df = self.raw_data

        for i, col in enumerate(timeseries_df.columns):
            if col in ["time", "index"]: continue
            print(str(self.type_data) + " " + str(self.loc_code) + " " + str(col) + " proceeding...               ", end = "\r")

            stl = STL(timeseries_df[col], period = period)
            stl_data = stl.fit()
            stl_data.plot()
            stl_data_trend = stl_data.trend
            stl_data_seasonal = stl_data.seasonal
            stl_data_residual = stl_data.resid

            self.trend = pd.concat(
                [self.trend, stl_data_trend],
                axis = 1
            )
            self.se8760 = pd.concat(
                [self.se8760, stl_data_seasonal],
                axis = 1
            )
            self.resid = pd.concat(
                [self.resid, stl_data_residual],
                axis = 1
            )

        print("")
                                      
    def save_newdata(self, key=""):
        self.trend.to_csv(f"./data_MSTL/MSTL{key}_{self.type_data}_{self.loc_code}_trend.csv")
        self.se24.to_csv(f"./data_MSTL/MSTL{key}_{self.type_data}_{self.loc_code}_se24.csv")
        self.se360.to_csv(f"./data_MSTL/MSTL{key}_{self.type_data}_{self.loc_code}_se360.csv")
        self.se8760.to_csv(f"./data_MSTL/MSTL{key}_{self.type_data}_{self.loc_code}_se8760.csv")
        self.resid.to_csv(f"./data_MSTL/MSTL{key}_{self.type_data}_{self.loc_code}_resid.csv")
            

def dataset_stl(code):
    assert code in ["1018640", "1018662", "1018680", "1018683"]
    
    matching_dict = {
        "1018640": ["402", "403", "413"],
        "1018662": ["403", "413", "421"],
        "1018680": ["415", "510", "889"], 
        "1018683": ["415", "510", "889"]
    }
    
    aws_needed = matching_dict[code]
    
    stl_aws_df1 = dataMSTL("climate", aws_needed[0])
    stl_aws_df2 = dataMSTL("climate", aws_needed[1])
    stl_aws_df3 = dataMSTL("climate", aws_needed[2])
    stl_lvl_df = dataMSTL("lvl", code)
    
    stl_aws_df1.my_stl()
    stl_aws_df2.my_stl()
    stl_aws_df3.my_stl()
    stl_lvl_df.my_stl()
    
    stl_aws_df1.save_newdata()
    stl_aws_df2.save_newdata()
    stl_aws_df3.save_newdata()
    stl_lvl_df.save_newdata()