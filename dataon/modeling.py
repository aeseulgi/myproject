import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy import stats

from sklearn.metrics import mean_squared_error 

import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

import tensorflow as tf
from tensorflow import keras

from pyFTS.common import Transformations, Util as cUtil, Membership as mf
from pyFTS.benchmarks import benchmarks as bchmk, Util as bUtil, Measures
from pyFTS.partitioners import CMeans, Grid, FCM, Huarng, Entropy, Util as pUtil
from pyFTS.models.multivariate import common, variable, mvfts
from pyFTS.models import hofts
from pyFTS.models import chen

import math
from joblib import dump, load

class predict_model:
    
    def __init__(self, df, code):
        self.type_dict = {'trend': 0, 'seasonal': 1, 'residual': 2}
        
        self.dataset = df
        self.code = code # 수위 예측 지점 코드
        
        self.trend_model = 0
        
        self.scaler_X = []
        self.scaler_y = []
        self.seasonal_model = 0
        
        self.df = []
    
    @staticmethod
    def SMAPE(y_test, y_pred):
        y_test = y_test.to_numpy(); y_pred = y_pred.to_numpy()
        return np.mean((np.abs(y_test-y_pred))/(np.abs(y_test)+np.abs(y_pred)))*100

    @staticmethod
    def RMSE(y_test, y_pred):
        return np.sqrt(mean_squared_error(y_test, y_pred))
    
    @staticmethod
    def SMAPE_np(y_test, y_pred):
        return np.mean((np.abs(y_test-y_pred))/(np.abs(y_test)+np.abs(y_pred)))*100
    
    @staticmethod
    def RMSE_np(y_test, y_pred):
        return np.mean((y_test - y_pred)**2)
        
    def fit_VAR_model(self, my_type, is_final, lag = 15, step = 3):
        given_data = self.dataset[self.type_dict[my_type]] # trend
        
        least_rmse = 1000

        for idx, block_data in enumerate(given_data):
            X_train = block_data[0]
            y_train = block_data[1]
            X_valid = block_data[2]
            y_valid = block_data[3]
            
            SMAPE_sum = 0
            RMSE_sum = 0
            
            df_train = pd.merge(y_train, X_train, left_index=True, right_index=True, how='inner')
            df_valid = pd.merge(y_valid, X_valid, left_index=True, right_index=True, how='inner')

            print(f"Modeling {my_type} component... SPLIT %d"%idx, end = "\r")
            premodel = VAR(df_train)
            model = premodel.fit(lag)

            df_valid = pd.concat([df_train[-lag:], df_valid])
            
            one_hour_pred = []
            one_hour_real = []
            two_hour_pred = []
            two_hour_real = []
            thr_hour_pred = []
            thr_hour_real = []
            

            for i in range(0, len(df_valid)-lag-step):
                lagged_values = df_valid.values[i: i+lag]
                forecast = pd.DataFrame(model.forecast(y = lagged_values, steps = step), columns=df_train.columns)
                one_hour_pred.append(forecast.iloc[0, [0]])
                two_hour_pred.append(forecast.iloc[1, [0]])
                thr_hour_pred.append(forecast.iloc[2, [0]])
                one_hour_real.append(df_valid.iloc[i+lag, [0]])
                two_hour_real.append(df_valid.iloc[i+lag+1, [0]])
                thr_hour_real.append(df_valid.iloc[i+lag+2, [0]])
                SMAPE_sum += self.SMAPE(forecast.iloc[:step, [0]], df_valid.iloc[i+lag: i+step+lag, [0]])
                RMSE_sum += self.RMSE(forecast.iloc[:step, [0]], df_valid.iloc[i+lag: i+step+lag, [0]]) 
            
            plt.plot(one_hour_real, label = "actual")
            plt.plot(one_hour_pred, label = "prediction")
            plt.show()
            plt.plot(two_hour_real, label = "actual")
            plt.plot(two_hour_pred, label = "prediction")
            plt.show()
            plt.plot(thr_hour_real, label = "actual")
            plt.plot(thr_hour_pred, label = "prediction")
            plt.show()
            
            print(f"TREND SPLIT {idx}: SMAPE avg {SMAPE_sum / (len(df_valid) - lag)}, RMSE avg {RMSE_sum / (len(df_valid) - lag)}")
            
            if least_rmse > RMSE_sum / (len(df_valid) - lag):
                least_rmse = RMSE_sum / (len(df_valid) - lag)
                best_model = model
                
        if is_final == 1:
            dump(best_model, f'../{my_type}_{self.code}_model_final.joblib')
        else:
            dump(best_model, f'../{my_type}_{self.code}_model.joblib')
        
    def eval_VAR_model(self, my_type):
        df = self.dataset
        
        lag = 15; step = 3
        
        model = load(f'../{my_type}_{self.code}_model_final.joblib')
        
        given_data = df[self.type_dict[my_type]]
        
        X_test = given_data[0][2]
        y_test = given_data[0][3]
        
        df_test = pd.merge(y_test, X_test, left_index=True, right_index=True, how='inner')
        
        pred = []
        
        for i in range(0, len(df_test)-lag-step):
            lagged_values = df_test.values[i: i+lag]
            forecast = pd.DataFrame(model.forecast(y = lagged_values, steps = step), columns=df_test.columns)
            pred.append([[forecast[0][0]], [forecast[0][1]], [forecast[0][2]]])
           
        return np.array(pred)


    def pred_VAR_model(self, my_type):
        
        df = self.dataset
        
        lag = 15; step = 3
        
        model = load(f'../{my_type}_{self.code}_model_final.joblib')
        
        given_data = df[self.type_dict[my_type]]
        
        X_test = given_data[0][0]
        y_test = given_data[0][1]
        
        df_test = pd.merge(y_test, X_test, left_index=True, right_index=True, how='inner')
        
        pred = []
        
        i = len(df_test)-lag
        lagged_values = df_test.values[i: i+lag]

        forecast = pd.DataFrame(model.forecast(y = lagged_values, steps = step), columns=df_test.columns)
        pred.append([[forecast[0][0]], [forecast[0][1]], [forecast[0][2]]])
            
        return np.array(pred)
        
    ###################         

    def make_dataset(self, X_df, y_df, time_step = 30, y_step = 3):
        new_X_df = []
        new_y_df = []
        X_df = pd.concat([X_df, y_df], axis=1)
        for i in range(len(X_df) - time_step - y_step + 1):
            new_X_df.append(np.array(X_df.iloc[i:i + time_step])) # i ~ i + time_step - 1
            new_y_df.append(np.array(y_df.iloc[i + time_step: i + time_step + y_step])) # i + time_step ~ i + time_step + y_step - 1
        return np.array(new_X_df), np.array(new_y_df)
    
    
    def calculate_metrics(self, true, pred):
        true = pd.DataFrame(true)
        pred = pd.DataFrame(pred)
        mae = (true - pred).abs().mean()
        mape = (true - pred).abs().div(true).mean() * 100
        rmse = np.sqrt(((true - pred) ** 2).mean())
        return {
            "mae": mae,
            "mape": mape,
            "rmse": rmse,
        }
    
    def Bi_Seq2_model(self, feature_num, time_step, y_step):

        model = keras.Sequential()
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(128, activation = 'tanh', input_shape = (time_step, feature_num), return_sequences = True, kernel_initializer='he_normal')))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(64, activation = 'tanh')))
        model.add(keras.layers.RepeatVector(y_step))
        model.add(keras.layers.LSTM(64, activation = 'tanh', return_sequences = True))
        model.add(keras.layers.LSTM(128, activation = 'tanh', return_sequences = True))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
        #model.compile(optimizer = 'nadam', loss = 'mse')
        #model.summary()

        return model
    
    def Bi_Seq2_fit(self, model, X_train, y_train, X_valid, y_valid, time_step = 8, patience = 50, epochs = 1000):

        model.compile(loss='mean_squared_error', optimizer = 'nadam')
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience)

        history = model.fit(X_train, y_train, 
                            epochs = epochs,
                            validation_data = (X_valid, y_valid), 
                            callbacks = [early_stop],
                            shuffle = False)
        return history
    
    def fit_Bi_Seq2(self, my_type, is_final, time_step = 30, y_step = 3, patience = 100, epochs = 1000):
        
        self.scaled_data = self.dataset[self.type_dict[my_type]]
        
        pred = []
        
        with tf.device('/device:GPU:0'):
            least_rmse = 1000
            for i in range(len(self.dataset[0])):
                (lstm_X_train, lstm_y_train) = self.make_dataset(self.scaled_data[i][0], self.scaled_data[i][1], time_step)
                (lstm_X_valid, lstm_y_valid) = self.make_dataset(self.scaled_data[i][2], self.scaled_data[i][3], time_step)

                model = self.Bi_Seq2_model(len(lstm_X_train[0][0]), time_step, y_step)

                history = self.Bi_Seq2_fit(model, lstm_X_train, lstm_y_train, lstm_X_valid, lstm_y_valid, time_step, patience, epochs)

                pred_temp = model.predict(lstm_X_valid)

                rmse = 0

                for j in range(y_step):
                    pred.append(pred_temp)
                    mat = self.calculate_metrics(lstm_y_valid[:, j], pred_temp[:, j])
                    print(mat)
                    rmse = rmse + mat['rmse'][0]

                    plt.plot(lstm_y_valid[:, j], label = 'actual')
                    plt.plot(pred_temp[:, j], label = 'prediction')
                        # print(self.calculate_metrics(self.inverse_min_max_scaling(self.scaler_y[i], lstm_y_valid[:, j]), self.inverse_min_max_scaling(self.scaler_y[i], pred_temp[:, j])))
                        # plt.plot(self.inverse_min_max_scaling(self.scaler_y[i], lstm_y_valid[:, j]), label = 'actual')
                        # plt.plot(self.inverse_min_max_scaling(self.scaler_y[i], pred_temp[:, j]), label = 'prediction')
                    plt.legend()
                    plt.show()

                rmse = rmse / y_step

                if rmse < least_rmse:
                    least_rmse = rmse
                    best_time_step = time_step
                    best_model = model
        
        if is_final == 1:
            best_model.save(f'../{my_type}_{self.code}_model_final.h5')
        else:
            best_model.save(f'../{my_type}_{self.code}_model.h5')
    
    def eval_Bi_Seq2(self, df, my_type): # 모델 평가를 위한 예측값을 도출하는 함수
        time_step = 30
        
        model = tf.keras.models.load_model(f'../{my_type}_{self.code}_model.h5')
        
        new_df = df[self.type_dict[my_type]]
        (lstm_X_test, lstm_y_test) = self.make_dataset(new_df[0][0], new_df[0][1], time_step)
        
        pred = model.predict(lstm_X_test)
        
        return pred
        
    def pred_Bi_Seq2(self, my_type): # 실제 예측을 위한 함수
        
        df = self.dataset
        
        time_step = 30
        
        model = tf.keras.models.load_model(f'../{my_type}_{self.code}_model_final.h5')
        new_df = df[1]
        lstm_X_test = pd.concat([new_df[0][0].iloc[-time_step:], new_df[0][1].iloc[-time_step:]], axis=1)
        (lstm_X_temp, lstm_y_temp) = self.make_dataset(new_df[0][0], new_df[0][1], time_step)

        lstm_X_test = np.array([lstm_X_test])

        return model.predict(lstm_X_test)
    
    
    def generate_generator(self, df):
        generator = dict()

        print("Generating the generator... lol")
        tested_data = []
        num_cols = len(df.columns)
        for i in range(num_cols):
            tested_data.append(df.iloc[:,i].to_numpy())

        tdiff = Transformations.Differential(1)

        dataset_names = ["pca%d"%i for i in range(1, num_cols)]
        dataset_names.append("waterlvl")

        print("Modeling the generator... ")

        partitioners_diff = {}


        for count, dataset_name in enumerate(dataset_names):
            print(count ,end = " ")
            dataset = tested_data[count]

            partitioner_diff = CMeans.CMeansPartitioner(data = dataset, npart=12, transformation = tdiff)
            partitioners_diff[dataset_name] = partitioner_diff

            model = chen.ConventionalFTS(partitioner = partitioners_diff[dataset_name])
            model.append_transformation(tdiff)
            model.fit(dataset, save_model=True, file_path=f'./generator_ANFIS/'+dataset_name)
            generator[dataset_names[count]] = model

        return generator
    
    def ANFIS_model_generator(self, train_data):
        diff = Transformations.Differential(1)
        explan_variable = []
        i = len(train_data.columns) - 1

        if i >= 1:
            va_pca1 = variable.Variable(
                "PCA1",
                data_label="pca1",
                alias="pca1",
                partitioner = CMeans.CMeansPartitioner,
                npart=12,
                data = train_data,
                transformation = diff
            )
            explan_variable.append(va_pca1)
        if i >= 2:
            va_pca2 = variable.Variable(
                "PCA2",
                data_label="pca2",
                alias="pca2",
                partitioner = CMeans.CMeansPartitioner,
                npart=12,
                data = train_data,
                transformation = diff
            )
            explan_variable.append(va_pca2)
        if i >= 3:
            va_pca3 = variable.Variable(
                "PCA3",
                data_label="pca3",
                alias="pca3",
                partitioner = CMeans.CMeansPartitioner,
                npart=12,
                data = train_data,
                transformation = diff
            )
            explan_variable.append(va_pca3)
        if i >= 4:
            va_pca4 = variable.Variable(
                "PCA4",
                data_label="pca4",
                alias="pca4",
                partitioner = CMeans.CMeansPartitioner,
                npart=12,
                data = train_data,
                transformation = diff
            )
            explan_variable.append(va_pca4)
        if i >= 5:
            va_pca5 = variable.Variable(
                "PCA5",
                data_label="pca5",
                alias="pca5",
                partitioner = CMeans.CMeansPartitioner,
                npart=12,
                data = train_data,
                transformation = diff
            )
            explan_variable.append(va_pca5)
        if i >= 6:
            va_pca5 = variable.Variable(
                "PCA6",
                data_label="pca6",
                alias="pca6",
                partitioner = CMeans.CMeansPartitioner,
                npart=12,
                data = train_data,
                transformation = diff
            )
            explan_variable.append(va_pca5)
        if i >= 7:
            va_pca5 = variable.Variable(
                "PCA7",
                data_label="pca7",
                alias="pca7",
                partitioner = CMeans.CMeansPartitioner,
                npart=12,
                data = train_data,
                transformation = diff
            )
            explan_variable.append(va_pca5)
        va_lvl = variable.Variable(
            "LVL",
            data_label="0",
            alias="0",
            partitioner = CMeans.CMeansPartitioner,
            npart=12,
            data = train_data,
            transformation = diff
        )
        explan_variable.append(va_lvl)
        
        method = mvfts.MVFTS
        model_lvl = method(
            explanatory_variables = [va_pca1, va_pca2, va_pca3, va_pca4, va_lvl],
            target_variable = va_lvl
        )

        model_lvl.fit(train_data)

        return model_lvl
    
    def eval_ANFIS(self, comp):
        comp = self.type_dict[comp]
        full_data = self.dataset[comp]
        
        for split in range(len(full_data)):
            train_data = pd.concat([full_data[split][0], full_data[split][1]], axis = 1)
            valid_data = pd.concat([full_data[split][2], full_data[split][3]], axis = 1)

            forecast_ahead1 = dict()
            forecasted2 = []
            forecast_ahead2 = dict()
            forecasted3 = []
            
            other_generator = self.generate_generator(train_data)
            
            colnames = ["pca%d"%(i+1) for i in range(len(train_data.columns)-1)]
            colnames.append("0")
            train_data.columns = colnames
            valid_data.columns = colnames
            
            
            model_lvl = self.ANFIS_model_generator(train_data)

            forecasted1 = model_lvl.predict(valid_data)

            # for i in range(len(valid_data)):
            for i in range(700, 750):
                df = valid_data.iloc[i:(i+1)]

                for colnum in range(len(df.columns) - 1):
                    forecast_ahead1["pca%d"%(colnum+1)] = other_generator["pca%d"%(colnum+1)].predict(df["pca%d"%(colnum+1)].to_numpy())
                forecast_ahead1["0"] = forecasted1[i]

                forecast_ahead1 = pd.DataFrame(forecast_ahead1, index = [0])
                model_lvl_prime = self.ANFIS_model_generator(train_data)
                forecasted2.append(model_lvl_prime.forecast(forecast_ahead1))

                for colnum in range(len(df.columns) - 1):
                    forecast_ahead2["pca%d"%(colnum+1)] = other_generator["pca%d"%(colnum+1)].predict(forecast_ahead1["pca%d"%(colnum+1)].to_numpy())
                forecast_ahead2["0"] = forecasted2[i]

                forecast_ahead2 = pd.DataFrame(forecast_ahead2, index = [0])
                model_lvl_prime = self.ANFIS_model_generator(train_data)
                forecasted3.append(model_lvl_prime.forecast(forecast_ahead1))
                
            # compare_valid_data = valid_data["0"].iloc[1:]
#             compare_valid_data = valid_data["0"].iloc[1:26]

#             print(f"1hour prediction for split{split}: {self.RMSE(compare_valid_data, forecasted1[:25])}")
#             print(f"2hour prediction for split{split}: {self.RMSE(compare_valid_data, forecasted2[:25])}")
#             print(f"3hour prediction for split{split}: {self.RMSE(compare_valid_data, forecasted3[:25])}")

            print([[forecasted1, forecasted2, forecasted3]])
            return [[forecasted1, forecasted2, forecasted3]]
        
    def pred_ANFIS(self, comp):
        comp = self.type_dict[comp]
        full_data_prime = self.dataset[comp]
        train_data = pd.concat([full_data_prime[0][0], full_data_prime[0][1]], axis = 1)
        valid_data = pd.concat([full_data_prime[0][2], full_data_prime[0][3]], axis = 1)
        full_data = pd.concat([train_data, valid_data], axis = 0).reset_index().drop(columns = ["index"])
        
        colnames = ["pca%d"%(i+1) for i in range(len(full_data.columns)-1)]
        colnames.append("0")
        full_data.columns = colnames

        last_data = full_data.iloc[-1:]
        
        other_generator = self.generate_generator(full_data)

        forecast_ahead1 = dict()
        forecasted2 = []
        forecast_ahead2 = dict()
        forecasted3 = []

        model_lvl = self.ANFIS_model_generator(full_data)

        forecasted1 = [model_lvl.predict(last_data)]

        for colnum in range(len(full_data.columns) - 1):
            forecast_ahead1["pca%d"%(colnum+1)] = other_generator["pca%d"%(colnum+1)].predict(last_data["pca%d"%(colnum+1)].to_numpy())
        forecast_ahead1["0"] = forecasted1[0]

        forecast_ahead1 = pd.DataFrame(forecast_ahead1, index = [0])
        model_lvl_prime = self.ANFIS_model_generator(full_data)
        forecasted2.append(model_lvl_prime.forecast(forecast_ahead1))

        for colnum in range(len(full_data.columns) - 1):
            forecast_ahead2["pca%d"%(colnum+1)] = other_generator["pca%d"%(colnum+1)].predict(last_data["pca%d"%(colnum+1)].to_numpy())
        forecast_ahead2["0"] = forecasted2[0]

        forecast_ahead2 = pd.DataFrame(forecast_ahead2, index = [0])
        model_lvl_prime = self.ANFIS_model_generator(full_data)
        forecasted3.append(model_lvl_prime.forecast(forecast_ahead1))
        
        return [[forecasted1, forecasted2, forecasted3]]
    
    
    def eval_model(self, pre): #pre: instance of preprocessing class
        
        df = self.dataset
        trend_pred = self.eval_VAR_model("trend")
        seasonal_pred = self.eval_ANFIS("seasonal")
        residual_pred = self.eval_VAR_model("residual")
        
        y_step = 3
        lag = 15
        time_step = 30
        limit = 50
        
        trend_pred = trend_pred[700:700+limit-lag+1]; print(trend_pred)
        seasonal_pred = seasonal_pred[lag - 1:limit]
        residual_pred = residual_pred[700:700+limit-lag+1]; print(residual_pred)
        
        y_pred = np.concatenate([trend_pred, seasonal_pred, residual_pred], axis = 2)
        
        y_pred = y_pred.transpose((1,2,0))
        
        for i in range(len(y_pred)):
        # for i in range(200)
            y_pred[i] = pre.inverse_minmax(y_pred[i])
        #y_pred = np.transpose(np.sum(y_pred, axis=2))
        y_pred = np.transpose(np.sum(y_pred, axis = 1))
        
        y_real_trend = np.array(df[0][0][1][lag:limit+1])
        y_real_seasonal = np.array(df[1][0][1][lag:limit+1])
        y_real_resid = np.array(df[2][0][1][lag:limit+1])
        
        y_real = np.concatenate([y_real_trend, y_real_seasonal, y_real_resid], axis = 1)
        for i in range(len(y_real)):
            y_real[i] = pre.inverse_minmax(y_real[i])
        y_real = np.transpose(np.sum(y_real, axis = 1))
        
        y_real = y_real[time_step + y_step - 1:]
        
        eval_mat = []
        
        for i in range(y_step):
            eval_mat.append(self.calculate_metrics(y_pred[:,i], y_real))
            
            plt.plot(y_real, label = 'actual')
            plt.plot(y_pred[:,i], label = 'prediction')
            plt.show()
            
        return eval_mat
    
    def pred_model(self, pre):
        
        trend_pred = self.pred_VAR_model("trend")
        seasonal_pred = self.pred_ANFIS("seasonal")
        residual_pred = self.pred_VAR_model("residual")
        
        y_pred = np.concatenate([trend_pred, seasonal_pred, residual_pred], axis = 2)
        
        y_pred = y_pred.transpose((1,2,0))
        
        for i in range(len(y_pred)):
            y_pred[i] = pre.inverse_minmax(y_pred[i])
        y_pred = np.transpose(np.sum(y_pred, axis = 1))
        
        print(y_pred)
        
        pd.DataFrame(y_pred.reshape(3,1)).to_csv(path_or_buf=f'./data_predict/{self.code}_pred.csv')
    