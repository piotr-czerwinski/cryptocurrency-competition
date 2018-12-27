import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from tqdm import tqdm_notebook
from ml_metrics import rmsle, rmse, msle
import xgboost as xgb
from statsmodels.tsa.api import Holt
from sklearn.linear_model import LinearRegression

def predict_with_holt(history, n_forecast = 48):    
    predictions = []
    fit1 = Holt(history).fit(optimized=True)
    predictions = fit1.forecast(n_forecast)
    return predictions

def validate_test(test):
    if len(test) != 79200:
        raise Exception
    if test.value.isnull().any():
        raise Exception
    if test[test.value<=0].value.any():
        raise Exception
    if test.id.values[0] != 425:
        raise Exception
    if test.id.values[-1] != 937349:
        raise Exception

def find_continous_series(train):
    for ts_index in tqdm_notebook(train.ts.unique(), desc='ts loop'):
        if not is_series_continous(train[train.ts==ts_index]):
            yield ts_index

def is_series_continous(series): #skaczemy, gdy tylko jedna godzina. Może zmiana czasu...
    proper_end = series.datetime.values[0] +  pd.to_timedelta(len(series)-1, unit='h')
    proper_end_minus_1 = series.datetime.values[0] +  pd.to_timedelta(len(series)-2, unit='h')
    proper_end_plus_1 = series.datetime.values[0] +  pd.to_timedelta(len(series), unit='h')
    last_val = pd.to_datetime(series.datetime.values[-1])
    return last_val == pd.to_datetime(proper_end) or last_val == pd.to_datetime(proper_end_minus_1) or last_val == pd.to_datetime(proper_end_plus_1)

def with_nan_series(train):
    for ts_index in tqdm_notebook(train.ts.unique(), desc='ts loop'):
        df_train_index = train[train.ts==ts_index]
        if df_train_index.value.isnull().any():
            yield ts_index

def transform_into_horizontal_time_series(df, history_size, column_names, prediction_size, ignored_tail_size = 0, lagged_windows_count = 1, window_lag = 10, linear_predictions_count = 0):
    lst_ts_featurized = []
    for ts_index in df.ts.unique():
        df_ts = df[df.ts==ts_index]
        if ignored_tail_size>0:
            df_ts = df_ts[:-ignored_tail_size]
            
        for window_index in range(0,lagged_windows_count):
            train_window = [ts_index]
            
            history_for_window = df_ts.value.values[-1 -prediction_size - window_index * window_lag : 
                                                    -1 - prediction_size - history_size - window_index * window_lag :
                                                    -1]

            train_window.extend(history_for_window)
            if len(history_for_window) < history_size:
                last_value_repetead = np.empty(history_size - len(history_for_window))
                last_value_repetead.fill(history_for_window[-1])
                train_window.extend(last_value_repetead)
            
            if linear_predictions_count > 0:
                model = LinearRegression()
                model.fit(np.arange(0, len(history_for_window)).reshape(-1, 1), history_for_window[::-1])
                linear_predictions = model.predict(np.arange(len(history_for_window), len(history_for_window) + linear_predictions_count).reshape(-1, 1))
                train_window.extend(linear_predictions)

            if prediction_size > 0:
                if window_index==0:
                    preds_for_window = df_ts.value.values[-prediction_size -window_index * window_lag : ]
                else:                                                      
                    preds_for_window = df_ts.value.values[-prediction_size -window_index * window_lag : 
                                                        -window_index * window_lag]    
                train_window.extend(preds_for_window)

            lst_ts_featurized.append(train_window)    
    
    return pd.DataFrame(lst_ts_featurized, columns=column_names)    

def decompose_prediction_to_vertical(df_horizontal_by_ts, df_target_vertical, features_desc, add_real_future_values = False):
    ls_real_values = []
    ls_xgb_prediction = []
    ls_xgb_prediction_rolling_8 = []
    ls_xgb_prediction_rolling_2 = []
    ls_dummy_last = []
    ls_gradient_xgb_r_8_last = []

    for ts_index in tqdm_notebook(df_target_vertical.ts.unique(), desc='ts loop'):
        ts_for_index = df_horizontal_by_ts[df_horizontal_by_ts.ts == ts_index]

        if add_real_future_values:
            ls_real_values.extend(ts_for_index[features_desc.future].values[0])

        ds_predictions_xgb = pd.Series(ts_for_index[features_desc.future_preds].values[0])
        ds_predictions_xgb_rolling_8 = ds_predictions_xgb.copy()
        ds_predictions_xgb_rolling_8[8:] = ds_predictions_xgb.rolling(8).mean()[8:] 
        ds_predictions_xgb_rolling_4 = ds_predictions_xgb.copy()
        ds_predictions_xgb_rolling_4[4:] = ds_predictions_xgb.rolling(4).mean()[4:] 

        ls_xgb_prediction.extend(ds_predictions_xgb)
        ls_xgb_prediction_rolling_8.extend(ds_predictions_xgb_rolling_8)
        ls_xgb_prediction_rolling_2.extend(ds_predictions_xgb_rolling_4)

        last_value_repetead = np.empty(len(ds_predictions_xgb))
        last_value_repetead.fill(ts_for_index[features_desc.history_lags[0]].values[0])    
        ls_dummy_last.extend(last_value_repetead)

        ls_gradient_xgb_r_8_last.extend(gradient_combination(ds_predictions_xgb_rolling_8, last_value_repetead))


    if add_real_future_values:
        df_target_vertical['real_values'] = ls_real_values
    df_target_vertical['xgb_prediction'] = ls_xgb_prediction
    df_target_vertical['xgb_prediction_rolling_8'] = ls_xgb_prediction_rolling_8
    df_target_vertical['xgb_prediction_rolling_4'] = ls_xgb_prediction_rolling_2
    df_target_vertical['dummy_last'] = ls_dummy_last
    df_target_vertical['gradient_xgb_r_8_last'] = ls_gradient_xgb_r_8_last   

def build_prediction_models(train, test, feats, targets, xgb_params, apply_log_transform):
    models = {}
    scores = []
    for (index,target) in enumerate(tqdm_notebook(targets, desc='ts loop')):
        model_feats = feats
        eval_set = [(train[model_feats].values, train[target].values), (test[model_feats].values, test[target].values)]

        model = XGBWrapper(xgb_params, early_stopping_rounds=20, eval_set=eval_set, verbose=0)
        if apply_log_transform:
            score = run_log_model(train, test, feats, target, model)    
        else:            
            score = run_model(train, test, model_feats, target, model)
        scores.append(score)
        models[target] = model

    #joblib.dump(models, '../models/model_pkeeee.pkl') 
    #print('Scores first: {}, mean: {}, last: {}'.format(np.min(scores), np.mean(scores), np.max(scores)))
    return models

def plot_series(ts_range, train, test, plots_max = 10):
    for ts_index in itertools.islice(ts_range, plots_max):
        plt.figure(figsize=(15, 5))
        plt.title('Forecast for ts{}'.format(ts_index))
        plt.plot(train[train.ts==ts_index].datetime, train[train.ts==ts_index].value, label='train') 
        plt.plot(test[test.ts==ts_index].datetime, test[test.ts==ts_index].value, label='forecast')
        plt.legend()

def plot_validation(ts_range, train, predictions, plots_max = 10, validation_size=48):
    for ts_index in itertools.islice(ts_range, plots_max):
        plt.figure(figsize=(15, 5))
        plt.title('Forecast for ts{}'.format(ts_index))
        plt.plot(train[train.ts==ts_index].datetime, train[train.ts==ts_index].value, label='train') 
        plt.plot(train[train.ts==ts_index].datetime[-validation_size:], predictions[predictions.ts==ts_index].value, label='forecast')
        plt.legend()

def gradient_combination(series1, series2):
    assert len(series1)==len(series2)
    weights = np.linspace(0,1,len(series1))
    return weights*series2 + (1-weights)*series1

class XGBWrapper:
    def __init__(self, xgb_params={}, early_stopping_rounds=10, eval_set=None, verbose=False, eval_metric='rmse'):
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_set = eval_set
        self.verbose = verbose

        self.model =xgb.XGBRegressor(**xgb_params)

    def fit(self, X, y):

        if self.eval_set is None:
            self.eval_set = [(X, y)]

        self.model.fit(X, y, verbose=self.verbose, eval_metric=self.eval_metric, \
            eval_set=self.eval_set, early_stopping_rounds=self.early_stopping_rounds)

    def predict(self, X):
        return self.model.predict(X)

def run_model(df_train, df_test, feats, target, model):
    y_train = df_train[target].values
    X_train = df_train[feats].values

    y_test = df_test[target].values
    X_test = df_test[feats].values

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred[ y_pred < 0 ] = 0
    score = rmsle(y_test, y_pred)

    return score


def run_log_model(df_train, df_test, feats, target, model):
    X_train = df_train[feats].values
    y_train = df_train[target].values
    y_log_train = np.log(y_train + 1)

    X_test = df_test[feats].values
    y_test = df_test[target].values

    model.fit(X_train, y_log_train)

    y_log_pred = model.predict(X_test)
    y_pred = np.exp(y_log_pred) - 1
    y_pred[ y_pred < 0 ] = 0
    score = rmsle(y_test, y_pred)

    return score

def build_linear_models(train, test, feats, targets, apply_log_transform):
    models = {}
    scores = []
    for (index,target) in enumerate(tqdm_notebook(targets, desc='ts loop')):
        model_feats = feats

        model = LinearRegression()
        if apply_log_transform:
            score = run_log_model(train, test, feats, target, model)    
        else:            
            score = run_model(train, test, model_feats, target, model)
        scores.append(score)
        models[target] = model

    return models