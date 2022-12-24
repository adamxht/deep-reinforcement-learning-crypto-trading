import streamlit as st
from DataMaster import get_latest, getFearAndGreedIndex, getTimeSeriesPrediction, get_state
from PIL import Image
import hydralit_components as hc
import pandas as pd
import datetime
from datetime import date
import yfinance as yf
from DQN import DuellingDQN
import torch as T
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error 

# Constants
STATE_SPACE = 132
DEVICE = 'cpu'
WINDOW_SIZE = 4
act_dict = {0:-1, 1:1, 2:0}

# Add logo 
@st.cache()
def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

@st.cache(allow_output_mutation=True)
def get_from_range(asset_code, start_date = None, end_date = None):
    frequency = "1d"
    df = yf.download([asset_code], start=start_date, end=end_date, interval=frequency)
    return df

# ---------------------------------------------------- App Starts Here ------------------------------------------------------------

# Page Icon
img = Image.open("./Images/Logo.png")
icon = Image.open("./Images/Icon.png")

# Page title
st.set_page_config(
    page_title="Crypto Trading Bot",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded")

# specify the primary menu definition
menu_data = [
        {'id': '', 'icon': "Home", 'label':""},
        {'id': 'Trade', 'icon': "far fa-chart-bar", 'label':"Trade"},
        {'id':'Technical indicators','icon':"fas fa-tachometer-alt",'label':"Technical indicators"},
        {'id': "Evaluate", 'icon': "far fa-address-book",'label':"Evaluate"}
]
over_theme = {'txc_inactive': '#FFFFFF','menu_background':'#262730','txc_active':'#000000','option_active':'#d1b765'}
menu_id = hc.nav_bar(menu_definition=menu_data,override_theme=over_theme)#, hide_streamlit_markers=False)

# Style Buttons
m = st.markdown("""
<style>
button:first-child {
    background-color: #000000;
    border: 5px solid #d1b765;
    font-color: #d1b765;
    height: 50px;
    font-size: 10px;
    }
button:first-child:hover {
    border: 5px solid yellow;
    font-color: #000000;
button:first-child:OnClick {
    background-color: #000000;
    border: 5px solid yellow;
    font-color: #000000;
}
</style>""", unsafe_allow_html=True)

# Load agent first
    # 114 
DQN114_PATH = "./Agents/Saving DQN/DQN-114/online.pt"
dqn114 = DuellingDQN(input_dim = STATE_SPACE)
dqn114.load_state_dict(T.load(DQN114_PATH)) 

# 740 
DQN740_PATH = "./Agents/Saving DQN/DQN-740/online.pt"
dqn740 = DuellingDQN(input_dim = STATE_SPACE)
dqn740.load_state_dict(T.load(DQN740_PATH)) 

if menu_id == "Trade":
    st.runtime.legacy_caching.clear_cache()
    # Show header
    original_title = '<p style="font-family:Sans-serif; color: #d1b765; font-size: 50px;">Automated Cryptocurrency Trading </p>'
    st.markdown(original_title, unsafe_allow_html=True)
    
    # Show readme
    readme = st.checkbox("READ ME")

    if readme:
        st.write(" This page allows you to see what the Agents think is the best decision for trading the selected Cryptos. However, it is best to avoid following the AI when it comes to trading Bitcoin, Cardano, or WAVES as they did not earn profit during the testing phase.")

    # Creating sidebar
    sideb = st.sidebar
    
    sideb.image(add_logo(logo_path="./Images/Logo.png", width=1000, height=500))
    sideb.header("Trading selections : ")

    agent = sideb.radio(label = "Agents : ", options = ["DQNv114", 'DQNv740'])

    sideb.write("")
    sideb.write("")
    asset_codes = sideb.selectbox(
        'Select the asset you want to trade : ',
         ['ETH-USD', 'DOGE-USD', 'BNB-USD', 'ADA-USD', 'WAVES-USD', 'BTC-USD']
     )
    sideb.write("")
    
    df = get_latest(asset_codes)
    df_last = df.tail(1).index
    last_date = pd.to_datetime(df_last).date
    last_date = last_date[0]
    
    data, unscaled_data= get_state(df, asset_codes)
    latest_price = unscaled_data["Close"][-1]
    predicted_price = unscaled_data["Prediction"][-1]
    pred_date = last_date + pd.Timedelta(days=1)
    
    if predicted_price > latest_price :
        trend = "Up"
        pred_color = "green"
    else : 
        trend = "Down"
        pred_color = "red"
   
    col1, col2 = st.columns(2)
    
    # Get Sentiments
    fng_data = unscaled_data["fNg_value"]
    avg_index = fng_data.mean()
    latest_index = fng_data[-1]
    latest_sentiment = ""
    if latest_index >= 0 and latest_index <20 :
        latest_sentiment = "Extreme Fear"
        sent_color = 'red'
    elif latest_index >= 20 and latest_index <40 :
        latest_sentiment = "Fear"
        sent_color = 'red'
    elif latest_index >= 40 and latest_index <60 :
        latest_sentiment = "Neutral"
        sent_color = 'white'
    elif latest_index >= 60 and latest_index <80 :
        latest_sentiment = "Greed"
        sent_color = 'green'
    else:
        latest_sentiment = "Extreme Greed"
        sent_color = 'green'
        
    fng_data=pd.DataFrame(fng_data)

    # Plot Closing price and predictions
    with col1:
        st.subheader(f' {asset_codes} last {len(unscaled_data)} days Closing Price ($)')
        st.write(f'Last Date : {last_date}          |        Current Price is : $ {latest_price:.2f}')
        pred_df = unscaled_data["Prediction"] 
        pred_df.index = pred_df.index + pd.Timedelta(days=1)
        chart_df = unscaled_data["Close"]
        chart_df.drop(chart_df.head(1).index,inplace=True) 
        compare = pd.concat([chart_df, pred_df], axis=1)
        realVals = compare.iloc[:-1]["Close"]
        predictedVals = compare.iloc[:-1]["Prediction"]
        rmse = mean_squared_error(realVals, predictedVals, squared = False)
        st.write(f' Predicted Price for {pred_date} is : :{pred_color}[$ {predicted_price:.2f} ({trend})] | RMSE : {rmse:.4f}')
        st.line_chart(compare)
    # Plot BTC fear and greed
    with col2:
        st.subheader(f'Bitcoin Fear and Greed Index last {len(unscaled_data)} days')
        st.write(f'{last_date} Sentiments : :{sent_color}[{latest_sentiment} ({latest_index})]')
        st.write(f"Average over {len(unscaled_data)} days : {avg_index}")
        st.line_chart(fng_data)
        
                
                
    # Agent's parameters (For demo purposes)
    initial_cap = 1000
    current_cap = 900
    running_cap = 800
    asset_inv = 1
    current_price = df.iloc[-1]['Adj Close']
    prev_act = 0
    
    data.reshape(1, -1)

    state = np.concatenate([data.T, [[current_cap/initial_cap,
                                                 running_cap/current_cap,
                                                 asset_inv * current_price/initial_cap,
                                                 prev_act]]], axis=0)

    state = state.reshape(-1, STATE_SPACE)
    state = T.from_numpy(state).float().to(DEVICE).view(1, -1)
    
    if agent == "DQNv114":
        actions =dqn740(state)
        actions = np.argmax(actions.cpu().data.numpy())
        action = act_dict[actions]
    elif agent == "DQNv740":
        actions =dqn114(state)
        actions = np.argmax(actions.cpu().data.numpy())
        action = act_dict[actions]
    
    # Print Agent's decision
    sideb.header(f"{agent}'s Suggestion :")
    # Buy
    if action == 1:
                yol= f"./HTML/Buy.html"
    # Hold
    elif action == 0:
                yol= f"./HTML/Hold.html"
    # Sell
    elif action == -1:
                yol= f"./HTML/Sell.html"
    f = open(yol,'r') 
    contents = f.read()
    f.close()
    
    sideb.markdown(contents, unsafe_allow_html=True)

    sideb.write("")
    sideb.write("")
    sideb.subheader(f' Trading Account Information :')
    sideb.write(f' - Initial Capital    :  $ {initial_cap}')
    sideb.write(f' - Current Capital :  $ {current_cap}')
    sideb.write(f' - Asset Invested  : {asset_inv} units')
    
elif menu_id == "Technical indicators":
    # Show header
    original_title = '<p style="font-family:Sans-serif; color: #d1b765; font-size: 50px;">Dashboard </p>'
    st.markdown(original_title, unsafe_allow_html=True)

    # Creating sidebar
    sideb = st.sidebar
    
    sideb.image(add_logo(logo_path="./Images/Logo.png", width=1000, height=500))
    sideb.header("Dashboard selections : ")
    
    plot_chart = sideb.radio(label = "Plot Asset Closing Price chart? : ",  options=["Yes", "No"])
    
    if plot_chart == "Yes":
        
        asset_codes = sideb.selectbox(
        'Select the asset you want to plot : ',
         ['ETH-USD', 'DOGE-USD', 'BNB-USD', 'ADA-USD', 'WAVES-USD', 'BTC-USD']
         )
        
        from_begin = sideb.radio(label = "From the beginning : ",  options=["Yes", "No"])
        if from_begin == "Yes":
            df = get_from_range(asset_codes)
        else :
            earliest = sideb.date_input('start date', value = (date(2018, 1, 28)),min_value=(date(2018, 1, 28)))
            delta = date.today() - earliest
            end_date = date.today()
            start_date = date.today() - datetime.timedelta(days=delta.days) 
            df = get_from_range(asset_codes, start_date, end_date)
        start = df.index[0].date()
        end = df.index[-1].date()
        st.subheader(f"{asset_codes} Closing price from {start} to {end}")
        close_df = df["Close"]
        st.line_chart(close_df)
        
        plot_indicators = sideb.radio(label = "Plot Technical Indicators? : ",  options=["Yes", "No"])
        
        if plot_indicators == "Yes":
            indic_codes = sideb.radio(
            'Select the Technical Indicators you want to plot : ', options = 
             ['Relative Strength Index', 'Moving Average Convergence/Divergence', 'Bollinger Bands']
             )
            df["r-1"] = df["Adj Close"].pct_change(1) 
            if indic_codes == "Relative Strength Index":    
                rsi_lb = 5
                pos_gain = df["r-1"].where(df["r-1"] > 0, 0).ewm(rsi_lb).mean()
                neg_gain = df["r-1"].where(df["r-1"] < 0, 0).ewm(rsi_lb).mean()
                rs = np.abs(pos_gain/neg_gain)
                df["rsi"] = 100 * rs/(1 + rs)
                rsi_df = df["rsi"]
                st.subheader(f"{asset_codes} RSI {start} to {end}")
                st.line_chart(rsi_df)
            elif indic_codes == "Moving Average Convergence/Divergence":
                df["macd_26"] = df["r-1"].ewm(span=26, adjust=False).mean()
                df["macd_12"] = df["r-1"].ewm(span=12, adjust=False).mean()
                df["macd_bl"] = df["r-1"].ewm(span=9, adjust=False).mean()
                df["macd"] = df["macd_12"] - df["macd_26"]
                st.subheader(f"{asset_codes} MACD {start} to {end}")
                df_26 = df["macd_26"]
                df_12 = df["macd_12"]
                df_9 = df["macd_bl"]
                df_macd = df["macd"]
                macd_df  = pd.concat([df_26, df_12], axis = 1)
                macd_df = pd.concat([macd_df, df_9], axis = 1)
                macd_df = pd.concat([macd_df, df_macd], axis = 1)
                st.line_chart(macd_df)
            elif indic_codes == "Bollinger Bands":
                bollinger_lback = 10
                df["bollinger"] = df["r-1"].ewm(bollinger_lback).mean()
                df["low_bollinger"] = df["bollinger"] - 2 * df["r-1"].rolling(bollinger_lback).std()
                df["high_bollinger"] = df["bollinger"] + 2 * df["r-1"].rolling(bollinger_lback).std()
                st.subheader(f"{asset_codes} Bollinger Bands {start} to {end}")
                df_boll = df["bollinger"]
                df_loboll = df["low_bollinger"]
                df_hiboll = df["high_bollinger"]
                boll_df  = pd.concat([df_boll, df_loboll], axis = 1)
                boll_df = pd.concat([boll_df, df_hiboll], axis = 1)
                st.line_chart(boll_df)
                
elif menu_id == "Evaluate":
    # Show header
    big_title = '<p style="font-family:Sans-serif; color: #d1b765; font-size: 50px;text-align:center">Agents Performance</p>'
    st.markdown(big_title, unsafe_allow_html=True)
    
    # Show readme
    readme = st.checkbox("READ ME")

    if readme:
        st.write(" This page shows the testing result after training the DQN agents. The total accumulated profits earned by DQNv114 is \\$114 in all 6 assets in one year period, whereas DQNv740 earned $740 as the name implies. You can select the performance result of the Agents on any assets in the testing phase. The two graphs shown will be the amont of capital earned during the testing phase as well as the trading decisions made.")
        
    # Sidebar
    sideb = st.sidebar
    sideb.image(add_logo(logo_path="./Images/Logo.png", width=1000, height=500))
    asset_codes = sideb.selectbox(
        'Evaluate Asset : ',
         ['ETH-USD', 'DOGE-USD', 'BNB-USD', 'ADA-USD', 'WAVES-USD', 'BTC-USD']
         )

    st.header("Capital earned in one year")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("DQNv114")
        image = Image.open(f'./Images/DQNv114-{asset_codes}.png')
        image = image.resize((539, 237))
        st.image(image, caption=f'DQNv114 results on {asset_codes}')
    with col2:
        st.subheader("DQNv740")
        image = Image.open(f'./Images/DQNv740-{asset_codes}.png')
        image = image.resize((539, 237))
        st.image(image, caption=f'DQNv740 results on {asset_codes}')
        
    st.header("Trading Buy/Sell Signals")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("DQNv114")
        image = Image.open(f'./Images/DQNv114-{asset_codes}-sig.png')
        image = image.resize((539, 237))
        st.image(image, caption=f'DQNv114 signals on {asset_codes}')
    with col2:
        st.subheader("DQNv740")
        image = Image.open(f'./Images/DQNv740-{asset_codes}-sig.png')
        image = image.resize((539, 237))
        st.image(image, caption=f'DQNv740 signals on {asset_codes}')
        
    st.header("Summary")
    st.subheader("Profitable Assets: BNB-USD, ETH-USD, DOGE-USD")
    st.write("1. ETH-USD ( + $46.40 )")
    st.write("2. DOGE-USD ( + $735.15 )")
    st.write("3. BNB-USD ( + $639.77 )")
    st.subheader("Not Profitable Assets: BTC-USD, ADA-USD, WAVES-USD")
    st.write("1. ADA-USD ( - $55.70 )")
    st.write("2. WAVES-USD ( - $52.72 )")
    st.write("3. BTC-USD ( - $125.94 )")
    st.subheader("Best Agent based on Assets:")
    col1, col2, col3= st.columns(3)
    with col1:
        st.write("1. ETH-USD : DQNv114 ( + $46.40 )")
        st.write("2. DOGE-USD : DQNv114 ( + $735.15 )")
        st.write("3. BNB-USD : DQNv740 ( + $639.77 )")
    with col2:
        st.write("4. ADA-USD : DQNv114 ( - $55.70 )")
        st.write("5. WAVES-USD : DQNv740 ( - $52.72 )")
        st.write("6. BTC-USD : DQNv740 ( - $125.94 )")
        
        
                
elif menu_id == "":
    # Show header
    big_title = '<p style="font-family:Sans-serif; color: #d1b765; font-size: 50px;text-align:center">Welcome to Noid\'s Crypto Trading  Solution</p>'
    st.markdown(big_title, unsafe_allow_html=True)
    
    small_title = '<p style="font-family:Sans-serif; color: #FFFFFF; font-size: 20px;text-align:center">Use the Navigation Bar above to : </p>'
    st.markdown(small_title, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        image = Image.open(f'./Images/Trade.png')
        image = image.resize((412, 312))
        st.image(image)
        trade_title = '<p style="font-family:Sans-serif; color: #d1b765; font-size: 40px;text-align:center">Trade</p>'
        st.markdown(trade_title, unsafe_allow_html=True)
        trade_desc = '<p style="font-family:Sans-serif; color: #FFFFFF; font-size: 20px;text-align:center">Learn Crypto Trading Decisions suggested by our AI</p>'
        st.markdown(trade_desc, unsafe_allow_html=True)
    with col2:
        image = Image.open(f'./Images/Indicators.png')
        image = image.resize((412, 312))
        st.image(image)
        ind_title = '<p style="font-family:Sans-serif; color: #d1b765; font-size: 40px;text-align:center">Technical Indicators</p>'
        st.markdown(ind_title, unsafe_allow_html=True)
        ind_desc = '<p style="font-family:Sans-serif; color: #FFFFFF; font-size: 20px;text-align:center">Plot various Technical Indicators (RSI, MACD, BB)</p>'
        st.markdown(ind_desc, unsafe_allow_html=True)
    with col3:
        image = Image.open(f'./Images/Evaluate.png')
        image = image.resize((412, 312))
        st.image(image)
        eval_title = '<p style="font-family:Sans-serif; color: #d1b765; font-size: 40px;text-align:center">Evaluate</p>'
        st.markdown(eval_title, unsafe_allow_html=True)
        eval_desc = '<p style="font-family:Sans-serif; color: #FFFFFF; font-size: 20px;text-align:center">Evaluate the Trading Performance of our trained AI on the Test Set (one year)</p>'
        st.markdown(eval_desc, unsafe_allow_html=True)

                    

            
        
        



























