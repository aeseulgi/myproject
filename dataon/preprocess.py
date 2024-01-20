import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

if np.__version__ >= "1.20.0":
    print("Warning: wrong version of Numpy library, might cause some issues later. Please downgrade to below 1.20.0.")

class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]
            
class minmax():
    def __init__(self):
        pass
        
    def minmax_transform_fit(self, train_df, valid_df):

        scaler = MinMaxScaler()
        scaler = scaler.fit(train_df)

        train_scaled = scaler.transform(train_df)
        train_scaled = pd.DataFrame(train_scaled)

        valid_scaled = scaler.transform(valid_df)
        valid_scaled = pd.DataFrame(valid_scaled)

        return scaler, train_scaled, valid_scaled
    
    def minmax_transform(self, scaler, test_df):
        
        test_scaled = scaler.transform(test_df)
        test_scaled = pd.DataFrame(test_scaled)

        return test_scaled
    
    def inverse_minmax(self, scaler, df):
        return scaler.inverse_transform(df)
    
class myPCA():
    def __init__(self):
        pass

    def PCA_transform_fit(self, train_df, valid_df):
        n = len(train_df.columns)
        self.pca = PCA(n_components = n)
        pca_train = self.pca.fit_transform(train_df)
        self.pca_train_df = pd.DataFrame(pca_train, columns = [f"pca{num + 1}" for num in range(n)])

        eig = self.pca.explained_variance_
        ratio = self.pca.explained_variance_ratio_
        acc_ratio = ratio.cumsum()

        self.pca_info = pd.DataFrame({
            "eig for variance": eig,
            "acc. explained": acc_ratio},
            index = np.array([f"pca{num + 1}" for num in range(n)])
        )

        pca_valid = self.pca.transform(valid_df)
        self.pca_valid_df = pd.DataFrame(pca_valid, columns = [f"pca{num + 1}" for num in range(n)])

        return self.pca_train_df, self.pca_valid_df
    
    def PCA_transform(self, test_df):
        
        n = len(test_df.columns)
        pca_test = self.pca.transform(test_df)
        self.pca_test_df = pd.DataFrame(pca_test, columns = [f"pca{num + 1}" for num in range(n)])
        
        return self.pca_test_df

# eigenvalue for variance >= 0.70
# accumulated ratio of variance explained >= 0.80

    def set_params(self):
        variance = 0
        #explained = 0
        for i in range(len(self.pca_train_df.columns)):
            # print(i)
            # print(self.pca_info["eig for variance"][i])
            # print(self.pca_info["acc. explained"][i])
            variance += self.pca_info["eig for variance"][i]
            #explained += self.pca_info["acc. explained"][i]
            #print(explained)
            if (self.pca_info["acc. explained"][i] >= 0.80):
                self.param = i+1
                return self.param
        return len(self.pca_train_df.columns)-1    
    
class Preprocessing_minmax:
    
    matching_dict = {"1018640" : ["402", "403", "413"], "1018662": ["403", "413", "421"], "1018680": ["415", "510", "889"], "1018683": ["415", "510", "889"]}
    
    def __init__(self, code):
        
        self.code = code

    def merging(self):
        
        self.loc_code = self.matching_dict[self.code]
        
        resid_lvl = pd.read_csv(f"./data_MSTL/MSTL_lvl_{self.code}_resid.csv")
        #se24_lvl = pd.read_csv(f"./data_MSTL/MSTL3_lvl_{self.code}_se24.csv")
        #se360_lvl = pd.read_csv(f"./data_MSTL/MSTL3_lvl_{self.code}_se360.csv")
        se8760_lvl = pd.read_csv(f"./data_MSTL/MSTL_lvl_{self.code}_se8760.csv")
        trend_lvl = pd.read_csv(f"./data_MSTL/MSTL_lvl_{self.code}_trend.csv")
        
        resid_fir = pd.read_csv(f"./data_MSTL/MSTL_climate_{self.loc_code[0]}_resid.csv")
        #se24_fir = pd.read_csv(f"./data_MSTL/MSTL3_climate_{self.loc_code[0]}_se24.csv")
        #se360_fir = pd.read_csv(f"./data_MSTL/MSTL3_climate_{self.loc_code[0]}_se360.csv")
        se8760_fir = pd.read_csv(f"./data_MSTL/MSTL_climate_{self.loc_code[0]}_se8760.csv")
        trend_fir = pd.read_csv(f"./data_MSTL/MSTL_climate_{self.loc_code[0]}_trend.csv")
        
        resid_sec = pd.read_csv(f"./data_MSTL/MSTL_climate_{self.loc_code[1]}_resid.csv")
        #se24_sec = pd.read_csv(f"./data_MSTL/MSTL3_climate_{self.loc_code[1]}_se24.csv")
        #se360_sec = pd.read_csv(f"./data_MSTL/MSTL_climate_{self.loc_code[1]}_se360.csv")
        se8760_sec = pd.read_csv(f"./data_MSTL/MSTL_climate_{self.loc_code[1]}_se8760.csv")
        trend_sec = pd.read_csv(f"./data_MSTL/MSTL_climate_{self.loc_code[1]}_trend.csv")
        
        resid_thr = pd.read_csv(f"./data_MSTL/MSTL_climate_{self.loc_code[2]}_resid.csv")
        #se24_thr = pd.read_csv(f"./data_MSTL/MSTL3_climate_{self.loc_code[2]}_se24.csv")
        #se360_thr = pd.read_csv(f"./data_MSTL/MSTL3_climate_{self.loc_code[2]}_se360.csv")
        se8760_thr = pd.read_csv(f"./data_MSTL/MSTL_climate_{self.loc_code[2]}_se8760.csv")
        trend_thr = pd.read_csv(f"./data_MSTL/MSTL_climate_{self.loc_code[2]}_trend.csv")
        
        resid_lvl.drop(["Unnamed: 0"], inplace = True, axis = 1)
        #se24_lvl.drop("Unnamed: 0", inplace = True, axis = 1)
        #se360_lvl.drop("Unnamed: 0", inplace = True, axis = 1)
        se8760_lvl.drop(["Unnamed: 0"], inplace = True, axis = 1)
        trend_lvl.drop(["Unnamed: 0"], inplace = True, axis = 1)
        
        resid_fir.drop(["Unnamed: 0"], inplace = True, axis = 1)
        # se24_fir.drop("Unnamed: 0", inplace = True, axis = 1)
        #se360_fir.drop("Unnamed: 0", inplace = True, axis = 1)
        se8760_fir.drop(["Unnamed: 0"], inplace = True, axis = 1)
        trend_fir.drop(["Unnamed: 0"], inplace = True, axis = 1)
        
        resid_sec.drop(["Unnamed: 0"], inplace = True, axis = 1)
        # se24_sec.drop("Unnamed: 0", inplace = True, axis = 1)
        #se360_sec.drop("Unnamed: 0", inplace = True, axis = 1)
        se8760_sec.drop(["Unnamed: 0"], inplace = True, axis = 1)
        trend_sec.drop(["Unnamed: 0"], inplace = True, axis = 1)
        
        resid_thr.drop(["Unnamed: 0"], inplace = True, axis = 1)
        # se24_thr.drop("Unnamed: 0", inplace = True, axis = 1)
        #se360_thr.drop("Unnamed: 0", inplace = True, axis = 1)
        se8760_thr.drop(["Unnamed: 0"], inplace = True, axis = 1)
        trend_thr.drop(["Unnamed: 0"], inplace = True, axis = 1)
        
        resid_lvl.columns = ["lvl_re"]
        # se24_lvl.columns = ["lvl_se24"]
        #se360_lvl.columns = ["lvl_se360"]
        se8760_lvl.columns = ["lvl_se8760"]
        trend_lvl.columns = ["lvl_tr"]
        
        #print(resid_fir)
        print(self.loc_code[0])
        
        resid_fir.columns = ["fir_re_temp", "fir_re_pre", "fir_re_ver", "fir_re_hor"]
        # se24_fir.columns = ["fir_se24_temp", "fir_se24_pre", "fir_se24_ver", "fir_se24_hor"]
        #se360_fir.columns = ["fir_se360_temp", "fir_se360_pre", "fir_se360_ver", "fir_se360_hor"]
        se8760_fir.columns = ["fir_se8760_temp", "fir_se8760_pre", "fir_se8760_ver", "fir_se8760_hor"]
        trend_fir.columns = ["fir_tr_temp", "fir_tr_pre", "fir_tr_ver", "fir_tr_hor"]
        
        resid_sec.columns = ["sec_re_temp", "sec_re_pre", "sec_re_ver", "sec_re_hor"]
        # se24_sec.columns = ["sec_se24_temp", "sec_se24_pre", "sec_se24_ver", "sec_se24_hor"]
        #se360_sec.columns = ["sec_se360_temp", "sec_se360_pre", "sec_se360_ver", "sec_se360_hor"]
        se8760_sec.columns = ["sec_se8760_temp", "sec_se8760_pre", "sec_se8760_ver", "sec_se8760_hor"]
        trend_sec.columns = ["sec_tr_temp", "sec_tr_pre", "sec_tr_ver", "sec_tr_hor"]
        
        resid_thr.columns = ["thr_re_temp", "thr_re_pre", "thr_re_ver", "thr_re_hor"]
        # se24_thr.columns = ["thr_se24_temp", "thr_se24_pre", "thr_se24_ver", "thr_se24_hor"]
        #se360_thr.columns = ["thr_se360_temp", "thr_se360_pre", "thr_se360_ver", "thr_se360_hor"]
        se8760_thr.columns = ["thr_se8760_temp", "thr_se8760_pre", "thr_se8760_ver", "thr_se8760_hor"]
        trend_thr.columns = ["thr_tr_temp", "thr_tr_pre", "thr_tr_ver", "thr_tr_hor"]
        
        
        # self.merged_df = pd.concat([resid_lvl, se24_lvl, se360_lvl, se8760_lvl, trend_lvl,
        #                            resid_fir, se24_fir, se360_fir, se8760_fir, trend_fir,
        #                            resid_sec, se24_sec, se360_sec, se8760_sec, trend_sec,
        #                            resid_thr, se24_thr, se360_thr, se8760_thr, trend_thr], axis = 1)
            
        self.merged_df = pd.concat([resid_lvl, se8760_lvl, trend_lvl,
                                   resid_fir, se8760_fir, trend_fir,
                                   resid_sec, se8760_sec, trend_sec,
                                   resid_thr, se8760_thr, trend_thr], axis = 1)
        
        self.merged_df = self.merged_df.dropna()

        #merged_df.head(30)
        
    def make_ind_set(self, num):
        blocking_instance = BlockingTimeSeriesSplit(n_splits = num)
        split_data = blocking_instance.split(self.merged_df)
        ind_set = []
        for i in range(num):
            ind_set += next(split_data)
        return ind_set
        
    def do_blocking(self):
        (self.full_data, self.full_data_test) = self.blocking(8)
        (self.full_data_final_test, _) = self.blocking(1)
        
    def do_minmax_and_pca(self):
        self.minmax_and_pca(8)
        
    def blocking(self, num):
        
        ind_set = self.make_ind_set(num)
        
        # trend = []; seasonal_24 = []; seasonal_360 = []; seasonal_8760 = []; residual = []
        trend = []; seasonal_8760 = []; residual = []
        trend_test = []; seasonal_8760_test = []; residual_test = []

        for i, ind in enumerate(ind_set):
            if i%2 == 0:
                train_df = self.merged_df.loc[ind]
                train_df.reset_index()
                # x_train = train_df.drop(['lvl_tr', 'lvl_se24', 'lvl_se360', 'lvl_se8760', 'lvl_re'], axis=1)
                x_train = train_df.drop(['lvl_tr', 'lvl_se8760', 'lvl_re'], axis=1)
            
                x_train_tr = train_df[[
                    "fir_tr_temp", "fir_tr_pre", "fir_tr_ver", "fir_tr_hor",
                    "sec_tr_temp", "sec_tr_pre", "sec_tr_ver", "sec_tr_hor",
                    "thr_tr_temp", "thr_tr_pre", "thr_tr_ver", "thr_tr_hor"
                ]]
                # x_train_se24 = train_df[[
                #     "fir_se24_temp", "fir_se24_pre", "fir_se24_ver", "fir_se24_hor",
                #     "sec_se24_temp", "sec_se24_pre", "sec_se24_ver", "sec_se24_hor",
                #     "thr_se24_temp", "thr_se24_pre", "thr_se24_ver", "thr_se24_hor"
                # ]]
                # x_train_se360 = train_df[[
                #     "fir_se360_temp", "fir_se360_pre", "fir_se360_ver", "fir_se360_hor",
                #     "sec_se360_temp", "sec_se360_pre", "sec_se360_ver", "sec_se360_hor",
                #     "thr_se360_temp", "thr_se360_pre", "thr_se360_ver", "thr_se360_hor"
                # ]]
                x_train_se8760 = train_df[[
                    "fir_se8760_temp", "fir_se8760_pre", "fir_se8760_ver", "fir_se8760_hor",
                    "sec_se8760_temp", "sec_se8760_pre", "sec_se8760_ver", "sec_se8760_hor",
                    "thr_se8760_temp", "thr_se8760_pre", "thr_se8760_ver", "thr_se8760_hor"
                ]]
                x_train_re = train_df[[
                    "fir_re_temp", "fir_re_pre", "fir_re_ver", "fir_re_hor",
                    "sec_re_temp", "sec_re_pre", "sec_re_ver", "sec_re_hor",
                    "thr_re_temp", "thr_re_pre", "thr_re_ver", "thr_re_hor"
                ]]
                y_train_tr = train_df[['lvl_tr']]
                # y_train_se24 = train_df[['lvl_se24']]
                # y_train_se360 = train_df[['lvl_se360']]
                y_train_se8760 = train_df[['lvl_se8760']]
                y_train_re = train_df[['lvl_re']]
            else:
                valid_df = self.merged_df.loc[ind]
                valid_df.reset_index()
                
                #x_valid = valid_df.drop(['lvl_tr', 'lvl_se24', 'lvl_se360', 'lvl_se8760', 'lvl_re'], axis=1)
                x_valid = valid_df.drop(['lvl_tr', 'lvl_se8760', 'lvl_re'], axis=1)
                
                x_valid_tr = valid_df[[
                    "fir_tr_temp", "fir_tr_pre", "fir_tr_ver", "fir_tr_hor",
                    "sec_tr_temp", "sec_tr_pre", "sec_tr_ver", "sec_tr_hor",
                    "thr_tr_temp", "thr_tr_pre", "thr_tr_ver", "thr_tr_hor"
                ]]
                # x_valid_se24 = valid_df[[
                #     "fir_se24_temp", "fir_se24_pre", "fir_se24_ver", "fir_se24_hor",
                #     "sec_se24_temp", "sec_se24_pre", "sec_se24_ver", "sec_se24_hor",
                #     "thr_se24_temp", "thr_se24_pre", "thr_se24_ver", "thr_se24_hor"
                # ]]
                # x_valid_se360 = valid_df[[
                #     "fir_se360_temp", "fir_se360_pre", "fir_se360_ver", "fir_se360_hor",
                #     "sec_se360_temp", "sec_se360_pre", "sec_se360_ver", "sec_se360_hor",
                #     "thr_se360_temp", "thr_se360_pre", "thr_se360_ver", "thr_se360_hor"
                # ]]
                x_valid_se8760 = valid_df[[
                    "fir_se8760_temp", "fir_se8760_pre", "fir_se8760_ver", "fir_se8760_hor",
                    "sec_se8760_temp", "sec_se8760_pre", "sec_se8760_ver", "sec_se8760_hor",
                    "thr_se8760_temp", "thr_se8760_pre", "thr_se8760_ver", "thr_se8760_hor"
                ]]
                x_valid_re = valid_df[[
                    "fir_re_temp", "fir_re_pre", "fir_re_ver", "fir_re_hor",
                    "sec_re_temp", "sec_re_pre", "sec_re_ver", "sec_re_hor",
                    "thr_re_temp", "thr_re_pre", "thr_re_ver", "thr_re_hor"
                ]]
                y_valid_tr = valid_df[['lvl_tr']]
                # y_valid_se24 = valid_df[['lvl_se24']]
                # y_valid_se360 = valid_df[['lvl_se360']]
                y_valid_se8760 = valid_df[['lvl_se8760']]
                y_valid_re = valid_df[['lvl_re']]
                
                if (i == 2 * num - 1 and i > 1):
                    x_test_tr = pd.concat([x_train_tr, x_valid_tr], ignore_index=True)
                    y_test_tr = pd.concat([y_train_tr, y_valid_tr], ignore_index=True)
                    x_test_se8760 = pd.concat([x_train_se8760, x_valid_se8760], ignore_index=True)
                    y_test_se8760 = pd.concat([y_train_se8760, y_valid_se8760], ignore_index=True)
                    x_test_re = pd.concat([x_train_re, x_valid_re], ignore_index=True)
                    y_test_re = pd.concat([y_train_re, y_valid_re], ignore_index=True)
                    trend_test.append([x_test_tr, y_test_tr])
                    seasonal_8760_test.append([x_test_se8760, y_test_se8760])
                    residual_test.append([x_test_re, y_test_re])
                    
                else:
                    trend.append([x_train_tr, y_train_tr, x_valid_tr, y_valid_tr])
                    # seasonal_24.append([x_train_se24, y_train_se24, x_valid_se24, y_valid_se24])
                    # seasonal_360.append([x_train_se360, y_train_se360, x_valid_se360, y_valid_se360])
                    seasonal_8760.append([x_train_se8760, y_train_se8760, x_valid_se8760, y_valid_se8760])
                    residual.append([x_train_re, y_train_re, x_valid_re, y_valid_re]) 
        
        # self.full_data = [trend, seasonal_24, seasonal_360, seasonal_8760, residual]
        full_data = [trend, seasonal_8760, residual]
        full_data_test = [trend_test, seasonal_8760_test, residual_test]
        
        return (full_data, full_data_test)
        
        
        #self.full_data_final_test = [trend_test, seasonal_8760_test, residual_test]
    
    def minmax_and_pca(self, num):
        
        self.pca_full_data = self.full_data
        self.pca_full_data_test = self.full_data_test
        self.pca_full_data_final_test = self.full_data_final_test
        
        self.minmax = minmax()
        
        self.scaler_X = []
        self.scaler_y = []
        
        ind_set = self.make_ind_set(num)[:-2]
        
        for i in range(len(self.full_data)):
            
            input_df0 = pd.DataFrame([])
            input_df1 = pd.DataFrame([])
            input_df2 = pd.DataFrame([])
            input_df3 = pd.DataFrame([])
            
            input_df0_test = pd.DataFrame([])
            input_df1_test = pd.DataFrame([])
            
            input_df0_final_test = pd.DataFrame([])
            input_df1_final_test = pd.DataFrame([])
            input_df2_final_test = pd.DataFrame([])
            input_df3_final_test = pd.DataFrame([])
            
            for j in range(len(self.full_data[0])):
                input_df0 = pd.concat([input_df0, self.full_data[i][j][0]]) # x train
                input_df1 = pd.concat([input_df1, self.full_data[i][j][1]]) # y train
                input_df2 = pd.concat([input_df2, self.full_data[i][j][2]]) # x valid
                input_df3 = pd.concat([input_df3, self.full_data[i][j][3]]) # y valid
                
            input_df0_test = self.full_data_test[i][0][0] # x test
            input_df1_test =self.full_data_test[i][0][1] # y test
                
            input_df0_final_test = self.full_data_final_test[i][0][0] 
            input_df1_final_test = self.full_data_final_test[i][0][1] 
            input_df2_final_test = self.full_data_final_test[i][0][2] 
            input_df3_final_test = self.full_data_final_test[i][0][3] 
            
            (scaler, df0, df2) = self.minmax.minmax_transform_fit(input_df0, input_df2) # x
            self.scaler_X.append(scaler)
            
            df0_test = self.minmax.minmax_transform(scaler, input_df0_test)
            df0_final_test = self.minmax.minmax_transform(scaler, input_df0_final_test)
            df2_final_test = self.minmax.minmax_transform(scaler, input_df2_final_test)
            
            (scaler, df1, df3) = self.minmax.minmax_transform_fit(input_df1, input_df3) # y
            self.scaler_y.append(scaler)
            
            df1_test = self.minmax.minmax_transform(scaler, input_df1_test)
            df1_final_test = self.minmax.minmax_transform(scaler, input_df1_final_test)
            df3_final_test = self.minmax.minmax_transform(scaler, input_df3_final_test)
            
            self.pca_instance = myPCA()
            x_train_pca, x_valid_pca = self.pca_instance.PCA_transform_fit(df0, df2)
            no = self.pca_instance.set_params()
            x_train_pca, x_valid_pca = x_train_pca[[f"pca{d+1}" for d in range(no)]], x_valid_pca[[f"pca{d+1}" for d in range(no)]]
            y_train_pca, y_valid_pca = (df1, df3)
            
            x_test_pca = self.pca_instance.PCA_transform(df0_test)
            x_test_pca = x_test_pca[[f"pca{d+1}" for d in range(no)]]
            x_final_train_pca = self.pca_instance.PCA_transform(df0_final_test)
            x_final_train_pca = x_final_train_pca[[f"pca{d+1}" for d in range(no)]]
            x_final_valid_pca = self.pca_instance.PCA_transform(df2_final_test)
            x_final_valid_pca = x_final_valid_pca[[f"pca{d+1}" for d in range(no)]]
            
            y_test_pca = df1_test
            y_final_train_pca = df1_final_test
            y_final_valid_pca = df3_final_test

            delta1 = len(ind_set[0])
            delta2 = len(ind_set[1])
            
            for k, ind in enumerate(ind_set):
                if k % 2 == 0:
                    self.pca_full_data[i][k//2][0] = x_train_pca.loc[[ind[i] - delta2 * k//2 for i in range(len(ind))]] # x train
                    self.pca_full_data[i][k//2][0].reset_index(inplace=True, drop = True)
                    self.pca_full_data[i][k//2][1] = y_train_pca.loc[[ind[i] - delta2 * k//2 for i in range(len(ind))]] # y train
                    self.pca_full_data[i][k//2][1].reset_index(inplace=True, drop = True)
                else:
                    self.pca_full_data[i][k//2][2] = x_valid_pca.loc[[ind[i] - delta1 * (k//2 + 1) for i in range(len(ind))]] # x valid
                    self.pca_full_data[i][k//2][2].reset_index(inplace=True, drop = True)
                    self.pca_full_data[i][k//2][3] = y_valid_pca.loc[[ind[i] - delta1 * (k//2 + 1) for i in range(len(ind))]] # y valid 
                    self.pca_full_data[i][k//2][3].reset_index(inplace=True, drop = True)
        
            self.pca_full_data_test[i][0][0] = x_test_pca
            self.pca_full_data_test[i][0][0].reset_index(inplace=True, drop = True)
            self.pca_full_data_test[i][0][1] = y_test_pca
            self.pca_full_data_test[i][0][1].reset_index(inplace=True, drop = True)
            
            self.pca_full_data_final_test[i][0][0] = x_final_train_pca
            self.pca_full_data_final_test[i][0][0].reset_index(inplace=True, drop = True)
            self.pca_full_data_final_test[i][0][1] = y_final_train_pca
            self.pca_full_data_final_test[i][0][1].reset_index(inplace=True, drop = True)
            self.pca_full_data_final_test[i][0][2] = x_final_valid_pca
            self.pca_full_data_final_test[i][0][2].reset_index(inplace=True, drop = True)
            self.pca_full_data_final_test[i][0][3] = y_final_valid_pca
            self.pca_full_data_final_test[i][0][3].reset_index(inplace=True, drop = True)
            
            
    def inverse_minmax(self, pred): # df may be a prediction vector (3d array)
        
        return_pred = pred
        
        for i in range(3):
            return_pred[i] = self.minmax.inverse_minmax(self.scaler_y[i], pred[i].reshape(1, -1))
        return return_pred 