# Library for getting the data and preparing it for training

import datetime
from datetime import date
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import requests
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
# from sklearn.externals import joblib

WINDOW_SIZE = 4

# This function collects the train and test data by downloading the asset price data using the yfinance dataset
def dataCollector(asset_codes):
    # Training date
    # Get four days before earliest for time series prediction
    earliest = date(2018, 1, 28) # This is 4 days before the earliest date where fear and greed index is available
    delta = date.today() - earliest
    train_end_date = date.today() - datetime.timedelta(days=365)
    train_start_date = date.today() - datetime.timedelta(days=delta.days) 
    # Testing date (one year)
    test_end_date = date.today()
    test_start_date = test_end_date - datetime.timedelta(days=368)
    frequency = "1d"

    for i in asset_codes : 
        df_train =  yf.download([i], start=train_start_date, end=train_end_date, interval=frequency)
        df_train.to_csv(f".//Datasets//Train//{i}_train.csv")

    for i in asset_codes : 
        df_test =  yf.download([i], start=test_start_date, end=test_end_date, interval=frequency)
        df_test.to_csv(f".//Datasets//Test//{i}_test.csv")
    return df_train, df_test

        
# This class will download the price data using Yahoo's API yfinance, at the same time, features will be engineered automatically and scaled
# Features include : 

class DataPreparer:

    def __init__(self, asset="BTC-USD", sets = "Train", timeframes=[1, 2, 5, 10, 20, 40]):
        self.asset = asset
        self.sets = sets
        self.timeframes = timeframes
        self.getData()
        
    def getFearAndGreedIndex(self):
       # 4 years * 365 = 1460
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        params = {'limit': '0', 'format': 'csv', 'date_format': 'kr'}
        response = session.get('https://api.alternative.me/fng/', params = params)
        if response.status_code == 200:
            print("Fear and Greed index OK\n")
            soup = bs(response.text,"lxml")
            allLines = soup.text.split('\n') 

            fng = pd.DataFrame(columns=['Date','fNg_value', 'sentiment'])
            earliest = len(allLines) - 5
            for i in range(4, earliest):
                string = allLines[i]
                lists = string.split(",")
                lists[0] = datetime.datetime.strptime(lists[0], '%Y-%m-%d')
                fng.loc[len(fng)] = lists
            fng = fng.set_index('Date')
        else:
            print("Fear and Greed index failed to download\n")

        fng.drop(columns = ['sentiment'], inplace = True)
        
        return fng
    
    # Get time series prediction using the trained predictor models inside the "Predictors" file
    def getTimeSeriesPrediction(self, df):

        print("Loading Predictor model")
        predictor = load_model(f'./Predictors/{self.asset}_Predictor.h5')
        df_close = df[["Close"]]
        dataset = df_close.values
        pred_data = []
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        inputs = dataset.reshape(-1,1)
        inputs  = scaler.transform(inputs)
        
        # Save Scaler
#         scaler_filename = "Scalers/PredictorScaler.save"
#         joblib.dump(scaler, scaler_filename) 

        for i in range(WINDOW_SIZE,inputs.shape[0]):
                pred_data.append(inputs[i-WINDOW_SIZE:i,0])

        pred_data = np.array(pred_data)
        pred_data = pred_data.reshape(-1, WINDOW_SIZE)
        print("Predicting Closing Prices")
        closing_price = predictor.predict(pred_data)
        closing_price = np.reshape(closing_price, (closing_price.shape[0], closing_price.shape[1]))
        closing_price = scaler.inverse_transform(closing_price)

        return closing_price
        

    def getData(self):        
        
        # Import data that was downloaded
        # -------------------------
        print(f"\nGetting and Preparing data for {self.asset}")
        asset = self.asset
        if self.sets == "Train":
            df = pd.read_csv(f".//Datasets//Train//{asset}_train.csv")
        else :
            df = pd.read_csv(f".//Datasets//Test//{asset}_test.csv")

            
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
#         df_spy.set_index('Date', inplace=True)
#         df_spy.index = pd.to_datetime(df.index)
        
        # Feature Engineering
        # -------------------------
        
        # Future Return - Not included in Observation Space.
        df["rf"] = df["Adj Close"].pct_change().shift(-1)

        
        # Returns and Trading Volume Changes
        for i in self.timeframes:
#             df_spy[f"spy_ret-{i}"] = df_spy["Adj Close"].pct_change(i)  
            
            df[f"r-{i}"] = df["Adj Close"].pct_change(i)      
            df[f"v-{i}"] = df["Volume"].pct_change(i)
    
        # Volatility
        for i in [5, 10, 20, 40]:
#             df_spy[f'spy_sig-{i}'] = np.log(1 + df_spy["spy_ret-1"]).rolling(i).std()
            
            df[f'sig-{i}'] = np.log(1 + df["r-1"]).rolling(i).std()
            

#--------------------------------------------------------------------------------------------------------------------------

#         df["r-1"] = df["Adj Close"].pct_change(1)     
            
        # Moving Average Convergence Divergence (MACD)
        df["macd_26"] = df["r-1"].ewm(span=26, adjust=False).mean()
        df["macd_12"] = df["r-1"].ewm(span=12, adjust=False).mean()
        df["macd_bl"] = df["r-1"].ewm(span=9, adjust=False).mean()
        df["macd"] = df["macd_12"] - df["macd_26"]

        # Relative Strength Indicator (RSI)
        rsi_lb = 5
        pos_gain = df["r-1"].where(df["r-1"] > 0, 0).ewm(rsi_lb).mean()
        neg_gain = df["r-1"].where(df["r-1"] < 0, 0).ewm(rsi_lb).mean()
        rs = np.abs(pos_gain/neg_gain)
        df["rsi"] = 100 * rs/(1 + rs)

#         # Bollinger Bands
        bollinger_lback = 10
        df["bollinger"] = df["r-1"].ewm(bollinger_lback).mean()
        df["low_bollinger"] = df["bollinger"] - 2 * df["r-1"].rolling(bollinger_lback).std()
        df["high_bollinger"] = df["bollinger"] + 2 * df["r-1"].rolling(bollinger_lback).std()

#         df = df.merge(df_spy[[f"spy_ret-{i}" for i in self.timeframes] + [f"spy_sig-{i}" for i in [5, 10, 20, 40]]], 
#                   how="left", right_index=True, left_index=True)
        
        # Get time series prediction using the trained predictor models inside the "Predictors" file
        
        predictions = self.getTimeSeriesPrediction(df)
        df = df.iloc[4:]
        df["Prediction"] = predictions
        
        # Get the fear and greed index using alternative.me API
        fng = self.getFearAndGreedIndex()

        # We can use join since both dataframes have identical indexes(dates
        df = df.join(fng)

        #  Higher Fear and Greed index(fNg) means better sentiment -> Higher price
        for index, rows in df.iterrows():
            if pd.isnull(rows["fNg_value"]):
                df.loc[index, "fNg_value"] = '50'

#         df['sentiment'] = df['sentiment'].astype('int32')
        df['fNg_value'] = df['fNg_value'].astype('float64')
        
        # Data Cleaning - removing null values
        for c in df.columns:
            df[c].interpolate('linear', limit_direction='both', inplace=True) # Does not remove null value in the first row
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # The whole dataset
        self.frame = df.copy()

        # Scale numerical features 
        self.scaled_frame = df
        self.scaler = StandardScaler()
        self.df_colToScale = df
        self.scaled_data = self.scaler.fit_transform(self.df_colToScale)
        self.scaled_data = pd.DataFrame(self.scaled_data, index=self.df_colToScale.index, columns=self.df_colToScale.columns)
        self.scaled_frame.update(self.scaled_data)
        self.scaled_frame.drop(columns = ['rf'], inplace = True)
        
        self.data = np.array(self.scaled_frame)
        
        # Save Scaler
#         scaler_filename = "Scalers/StateScaler.save"
#         joblib.dump(self.scaler, scaler_filename) 

        return
    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx, col_idx=None):
        if col_idx is None:
              return self.data[idx]
        elif col_idx < len(list(self.data.columns)):
              return self.data[idx][col_idx]
        else:
              raise IndexError

# ------------------------------------------ FOR DEPLOYMENT ----------------------------------------
# 40 days data to compute returns, volume and volatility
@st.cache(allow_output_mutation=True)
def get_latest(asset_code):
    start_date = date.today() - datetime.timedelta(days=44)
    end_date = date.today()
    frequency = "1d"
    df = yf.download([asset_code], start=start_date, end=end_date, interval=frequency)
    return df

# Get Fear and Greed Index
@st.cache(allow_output_mutation=True)
def getFearAndGreedIndex():
   # 4 years * 365 = 1460
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    params = {'limit': '0', 'format': 'csv', 'date_format': 'kr'}
    response = session.get('https://api.alternative.me/fng/', params = params)
    if response.status_code == 200:
        print("Fear and Greed index OK\n")
        soup = bs(response.text,"lxml")
        allLines = soup.text.split('\n') 

        fng = pd.DataFrame(columns=['Date','fNg_value', 'sentiment'])
        earliest = len(allLines) - 5
        for i in range(4, earliest):
            string = allLines[i]
            lists = string.split(",")
            lists[0] = datetime.datetime.strptime(lists[0], '%Y-%m-%d')
            fng.loc[len(fng)] = lists
        fng = fng.set_index('Date')
    else:
        print("Fear and Greed index failed to download\n")

    fng.drop(columns = ['sentiment'], inplace = True)

    return fng

# Get time series prediction using the trained predictor models inside the "Predictors" file
# @st.cache(allow_output_mutation=True) # will not refresh when website refreshes
def getTimeSeriesPrediction(df, asset_code):
    print("Loading Predictor model")
    predictor = load_model(f'./Predictors/{asset_code}_Predictor.h5')
    df_close = df[["Close"]]
    dataset = df_close.values
    pred_data = []
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    inputs = dataset.reshape(-1,1)
    inputs  = scaler.transform(inputs)

    for i in range(WINDOW_SIZE,inputs.shape[0]):
        pred_data.append(inputs[i-WINDOW_SIZE:i,0])

    pred_data = np.array(pred_data)
    pred_data = pred_data.reshape(-1, WINDOW_SIZE)
    print("Predicting Closing Prices")
    closing_price = predictor.predict(pred_data)
    closing_price = np.reshape(closing_price, (closing_price.shape[0], closing_price.shape[1]))
    closing_price = scaler.inverse_transform(closing_price)

    return closing_price

# Get latest data for deployment
@st.cache(allow_output_mutation=True)
def get_state(df_realTime, asset_code):

    for i in [1, 2, 5, 10, 20, 40]:
        df_realTime[f"r-{i}"] = df_realTime["Adj Close"].pct_change(i)      
        df_realTime[f"v-{i}"] = df_realTime["Volume"].pct_change(i)

    # Volatility
    for i in [5, 10, 20, 40]:
        df_realTime[f'sig-{i}'] = np.log(1 + df_realTime["r-1"]).rolling(i).std()

    # Moving Average Convergence Divergence (MACD)
    df_realTime["macd_26"] = df_realTime["r-1"].ewm(span=26, adjust=False).mean()
    df_realTime["macd_12"] = df_realTime["r-1"].ewm(span=12, adjust=False).mean()
    df_realTime["macd_bl"] = df_realTime["r-1"].ewm(span=9, adjust=False).mean()
    df_realTime["macd"] = df_realTime["macd_12"] - df_realTime["macd_26"]

    # Relative Strength Indicator (RSI)
    rsi_lb = 5
    pos_gain = df_realTime["r-1"].where(df_realTime["r-1"] > 0, 0).ewm(rsi_lb).mean()
    neg_gain = df_realTime["r-1"].where(df_realTime["r-1"] < 0, 0).ewm(rsi_lb).mean()
    rs = np.abs(pos_gain/neg_gain)
    df_realTime["rsi"] = 100 * rs/(1 + rs)

    # Bollinger Bands
    bollinger_lback = 10
    df_realTime["bollinger"] = df_realTime["r-1"].ewm(bollinger_lback).mean()
    df_realTime["low_bollinger"] = df_realTime["bollinger"] - 2 * df_realTime["r-1"].rolling(bollinger_lback).std()
    df_realTime["high_bollinger"] = df_realTime["bollinger"] + 2 * df_realTime["r-1"].rolling(bollinger_lback).std()

    # Get time series prediction using the trained predictor models inside the "Predictors" file
    predictions = getTimeSeriesPrediction(df_realTime, asset_code)
    df = df_realTime.iloc[4:]
    df["Prediction"] = predictions
    
    # Get the fear and greed index using alternative.me API
    fng = getFearAndGreedIndex()
    df = df.join(fng)
    
    for index, rows in df.iterrows():
        if pd.isnull(rows["fNg_value"]):
            df.loc[index, "fNg_value"] = '50'

    df['fNg_value'] = df['fNg_value'].astype('float64')
    
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)

    return scaled_df[-4:, :], df

@st.cache(allow_output_mutation=True)
def get_from_range(asset_code, start_date = None, end_date = None):
    frequency = "1d"
    df = yf.download([asset_code], start=start_date, end=end_date, interval=frequency)
    return df