# -*- coding: utf-8 -*-
"""ML Project

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/115kVTh1SJ1Z9sk15i9w03fboRNn3trLC
"""

import numpy as np
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import yfinance as yfin

from datetime import date
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight



from google.colab import drive
drive.mount('/content/drive')
os.chdir('/content/drive/My Drive/Colab Notebooks/ML/Project')

#Note for grader, this code was run locally due to issues with google colab, but for sake of completeness I have included it in the notebook

# Constants
START_YEAR = 5  # How many years back to fetch data for training
PREDICTION_WINDOW = 180  # 6 months (approx. 180 days)

S_AND_P_500 = [
    "A", "AAPL", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ",
    "AJG", "AKAM", "ALB", "ALGN", "ALK", "ALL", "ALLE", "AMAT", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN",
    "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK",
    "AXP", "AZO", "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BIIB", "BIO", "BK",
    "BKNG", "BKR", "BLK", "BMY", "BR", "BRO", "BSX", "BWA", "C", "CAG", "CAH", "CARR", "CAT", "CB",
    "CBOE", "CBRE", "CCL", "CDNS", "CDW", "CE", "CF", "CFG", "CHD", "CHRW", "CI", "CINF",
    "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP", "COST",
    "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO", "CSGP", "CSX", "CTAS", "CTLT", "CTRA", "CTS", "CVS", "CVX",
    "D", "DAL", "DD", "DE", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR",
    "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXC", "DXCM", "EA", "EBAY", "ECL", "ED",
    "EFX", "EIX", "EL", "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "ES", "ESS", "ETN", "ETR", "EVRG",
    "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FIS",
    "FITB", "FMC", "FOX", "FOXA", "FRT", "FTNT", "FTV", "GD", "GE", "GILD",
    "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN",
    "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUM",
    "HWM", "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU", "IP", "IPG", "IQV", "IR", "IRM",
    "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KEY", "KEYS", "KHC",
    "KIM", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "L", "LDOS", "LEG", "LEN", "LH", "LHX", "LIN", "LKQ",
    "LLY", "LMT", "LNC", "LNT", "LOW", "LRCX", "LUMN", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP",
    "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO",
    "MOS", "MPC", "MPWR", "MRK", "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH",
    "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE",
    "NVDA", "NVR", "NWL", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS",
    "OXY", "PARA", "PAYC", "PAYX", "PCAR", "PEG", "PENN", "PEP", "PFE", "PFG", "PG", "PGR", "PH",
    "PHM", "PKG", "PLD", "PM", "PNC", "PNR", "PNW", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC",
    "PVH", "PWR", "QCOM", "QRVO", "RCL", "REG", "REGN", "RF", "RHI", "RJF", "RL", "RMD", "ROK",
    "ROL", "ROP", "ROST", "RSG", "RTX", "SBAC", "SBUX", "SCHW", "SEE", "SHW", "SJM", "SLB",
    "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SWK", "SWKS", "SYF", "SYK",
    "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", "TJX", "TMO", "TMUS", "TPR",
    "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL", "UDR",
    "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VFC", "VLO", "VMC", "VNO", "VRSK", "VRSN",
    "VRTX", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WHR", "WM", "WMB",
    "WMT", "WRB", "WST", "WTW", "WY", "WYNN", "XEL", "XYL", "YUM", "ZBH", "ZBRA", "ZION", "ZTS"
]

DATA_DIR = "/content/drive/My Drive/Colab Notebooks/ML/Project/"
TIME_SERIES_DIR = os.path.join(DATA_DIR, "time_series_data/")
FUNDAMENTAL_DIR = os.path.join(DATA_DIR, "fundamentals_data/")
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_stock_data(stock, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    try:
        print(f"Fetching data for {stock}...")
        fetch_fundamental_data(stock)
        stock_data = yfin.Ticker(stock)
        stock_df = stock_data.history(start=start_date, end=end_date)

        # Calculate price 6 months later
        stock_df['Date Plus 6 Months'] = pd.Timestamp(start_date) + pd.DateOffset(months=6)

        # Function to get the price 6 months later
        def get_price_6_months_later(row):
            future_date = row['Date Plus 6 Months']
            try:
                future_price_df = stock_data.history(start=future_date, end=(pd.Timestamp(future_date)+pd.DateOffset(days=1)))
                if not future_price_df.empty:
                    return future_price_df['Close'].iloc[0]
                return np.nan
            except Exception as e:
                print(f"Error fetching price for {future_date}: {e}")
                return np.nan

        # Apply the function to calculate prices 6 months later
        stock_df['Price 6 Months Later'] = stock_df.apply(get_price_6_months_later, axis=1)

        # Calculate the percentage change
        stock_df['Price Change (%)'] = ((stock_df['Price 6 Months Later'] - stock_df['Close']) / stock_df['Close']) * 100

        # Classify stock
        stock_df['Label'] = stock_df.apply(classify_stock, axis=1)

        # Save the data to CSV
        stock_df.to_csv(f"{TIME_SERIES_DIR}/{stock}_data.csv", index=False)
        print(f"Saved {stock}_data.csv")
    except Exception as e:
        print(f"Failed to fetch data for {stock}: {e}")

def fetch_fundamental_data(stock):
    """Fetch fundamental data from Yahoo Finance."""
    try:
        print(f"Fetching fundamentals for {stock}...")
        stock_data = yfin.Ticker(stock)
        stock_fundamentals = stock_data.info

        summary = stock_data.get_recommendations_summary()
        analyst_rec = {
            'Strong Buy':  np.mean([summary['strongBuy'][0],summary['strongBuy'][1],summary['strongBuy'][2],summary['strongBuy'][3]]),
            'Buy': np.mean([summary['buy'][0],summary['buy'][1],summary['buy'][2],summary['buy'][3]]),
            'Hold': np.mean([summary['hold'][0],summary['hold'][1],summary['hold'][2],summary['hold'][3]]),
            'Sell': np.mean([summary['sell'][0],summary['sell'][1],summary['sell'][2],summary['sell'][3]]),
            'Strong Sell': np.mean([summary['strongSell'][0],summary['strongSell'][1],summary['strongSell'][2],summary['strongSell'][3]])
        }
        freeCashflow = stock_fundamentals.get('freeCashflow')
        totalDebt = stock_fundamentals.get('totalDebt')

        label = max(analyst_rec, key=analyst_rec.get)
        trailingPE = stock_fundamentals.get('trailingPE')
        eps = (stock_fundamentals.get('netIncomeToCommon')/stock_fundamentals.get('sharesOutstanding'))
        earningsGrowth = stock_fundamentals.get('earningsGrowth')
        priceToBook = stock_fundamentals.get('priceToBook')
        returnOnEquity = stock_fundamentals.get('returnOnEquity')
        priceToSales = stock_fundamentals.get('priceToSalesTrailing12Months')
        marketCap = stock_fundamentals.get('marketCap')
        debtToEquity = stock_fundamentals.get('debtToEquity')
        if totalDebt is not None and freeCashflow is not None:
            cashToDebt = freeCashflow/totalDebt
        else:
            cashToDebt = 0

        stock_f = {}
        stock_f['trailingPE'] = trailingPE
        stock_f['EPS'] = eps
        stock_f['earningsGrowth'] = earningsGrowth
        stock_f['priceToBook'] = priceToBook
        stock_f['returnOnEquity'] = returnOnEquity
        stock_f['priceToSales'] = priceToSales
        stock_f['marketCap'] = marketCap
        stock_f['debtToEquity'] = debtToEquity
        stock_f['cashToDebt'] = cashToDebt
        stock_f['label'] = label

        with open(f"{FUNDAMENTAL_DIR}/{stock}_fundamentals.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(stock_f.keys())
            writer.writerow(stock_f.values())
        print(f"Saved {stock}_fundamentals.csv")
    except Exception as e:
        print(f"Failed to fetch data for {stock}: {e}")

def classify_stock(row):
    """Classify stocks into strong buy, weak buy, etc."""
    if row['Price Change (%)'] > 10:
        return 'strongBuy'
    elif 5 < row['Price Change (%)'] <= 10:
        return 'buy'
    elif -5 <= row['Price Change (%)'] <= 5:
        return 'hold'
    elif -10 <= row['Price Change (%)'] < -5:
        return 'sell'
    else:
        return 'strongSell'



# Download data for S&P 500
start_time = '2020-01-01'
end_time = '2023-01-01'

print("Running")
print("Start: ", start_time)
print("End: ", end_time)
with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(lambda stock: fetch_stock_data(stock, start_time, end_time), S_AND_P_500)

def interpret_trailingPE(value):
    if value < 0:  # Losing Money
        return 0  # Negative PE is a bad sign
    elif 0 < value < 15:  # Strong Undervalued Range
        return 2  # Strong undervaluation with potential upside
    elif 15 <= value < 25:  # Reasonable Undervalued Range
        return 1.5  # Undervalued but not excessively cheap
    elif 25 <= value < 40:  # Fairly Valued Range
        return 1.2  # Fairly valued, closer to market average
    elif 40 <= value < 60:  # Overvalued Range
        return 0.8  # Slightly overpriced
    else:  # Highly Overpriced Range
        return 0.5  # Potentially overpriced, caution needed

def interpret_EPS(value):
    if value > 10:
        return 2  # Strong earnings
    elif value > 5:
        return 1.5  # Strong earnings
    elif value > 1.5:
        return 1  # Moderate earnings
    elif value > 0.5:
        return 0.5  # Slight earnings
    else:
        return 0  # Weak earnings

def interpret_earningsGrowth(value):
    if value > 0.5:
        return 1.5  # Strong growth
    elif -0.5 < value < 0.5:
        return 0.5  # Relative stagnation
    else:
        return 0  # Weak growth

def interpret_cashToDebt(cashToDebt):
    if cashToDebt > 2:
        return 2  # Company able to pay off debt instantly
    elif 2 > cashToDebt >= 1:
        return 1.5  # Financially healthy
    elif 1 > cashToDebt >= 0.5:
        return 1  # Moderate free cash flow
    elif 0.5 > cashToDebt >= 0:
        return 0.5  # Weak free cash flow
    else:
        return 0  # Company is unable to pay debt anytime soon

def interpret_priceToBook(value):
    if value < 1:
        return 2  # Undervalued relative to assets
    elif 1 <= value < 2:
        return 1.5  # Fairly valued relative to assets
    elif 2 <= value < 4:
        return 1  # Slightly overvalued relative to assets
    else:
        return 0.5  # Significantly overvalued relative to assets

def interpret_returnOnEquity(value):
    value = value * 100
    if value > 20:
        return 2  # Excellent profitability, high return on equity
    elif 10 <= value <= 20:
        return 1.5  # Good profitability
    elif 5 <= value < 10:
        return 1  # Acceptable profitability
    else:
        return 0.5  # Poor profitability

def interpret_priceToSales(value):
    if value < 1:
        return 2  # Undervalued stock relative to sales
    elif 1 <= value < 2:
        return 1.5  # Fairly valued stock relative to sales
    elif 2 <= value < 4:
        return 1  # Slightly overvalued stock relative to sales
    else:
        return 0.5  # Highly overpriced stock relative to sales

def interpret_marketCap(value):
    if value > 1000000000000:  # Large-cap stocks (over 1 trillion)
        return 2  # Large companies are generally safer
    elif 500000000000 <= value <= 1000000000000:  # Mid-cap stocks (500B to 1T)
        return 1.5  # Stable but more growth potential
    elif 10000000000 <= value < 500000000000:  # Small to mid-cap stocks (10B to 500B)
        return 1  # More risk but potential for growth
    else:
        return 0.5  # Small-cap stocks with higher risk but higher growth potential

def interpret_debtToEquity(value):
    if value < 0.5:
        return 2  # Low debt, strong financial stability
    elif 0.5 <= value < 1:
        return 1.5  # Moderate debt, manageable risk
    elif 1 <= value < 2:
        return 1  # High debt, more risk
    elif value == 0:
        return 0
    else:
        return 0.5  # Very high debt, high financial risk

fundamentals_dir = "fundamentals_data/"
time_series_dir = "time_series_data/"
num_classes = 5
fundamental_features = ['trailingPE', 'EPS', 'earningsGrowth',
                        'priceToBook', 'returnOnEquity', 'priceToSales',
                        'marketCap', 'debtToEquity', 'cashToDebt']
time_series_features = ['Open', 'Close']
label_column = 'label'

feature_weights = {
    'trailingPE': 2.0,       # High importance
    'EPS': 1.8,              # High importance
    'earningsGrowth': 2.5,   # Very high importance
    'priceToBook': 1.0,      # Medium importance
    'returnOnEquity': 1.5,   # High importance
    'priceToSales': 1.1,     # Medium importance
    'marketCap': 0.7,        # Lower importance
    'debtToEquity': 0.9,     # Lower importance
    'cashToDebt': 1.4        # High importance
}

def load_fundamental_data(fundamentals_dir):
    """Load and preprocess fundamental data for all tickers."""
    all_fundamentals = []
    interpretation_functions = {
        'trailingPE': interpret_trailingPE,
        'EPS': interpret_EPS,
        'earningsGrowth': interpret_earningsGrowth,
        'priceToBook': interpret_priceToBook,
        'returnOnEquity': interpret_returnOnEquity,
        'priceToSales': interpret_priceToSales,
        'marketCap': interpret_marketCap,
        'debtToEquity': interpret_debtToEquity,
        'cashToDebt': interpret_cashToDebt
    }
    for file in os.listdir(fundamentals_dir):
        if file.endswith(".csv"):
            print("File: ", file)
            ticker = file.split('_')[0]
            df = pd.read_csv(os.path.join(fundamentals_dir, file))
            df["Ticker"] = ticker
            df.replace("", np.nan, inplace=True)
            if df.isnull().values.any():
                print(f"Missing values found in {file}.")
            for feature in fundamental_features:
                if feature in df.columns:
                    df[feature] = df[feature].apply(interpretation_functions.get(feature, lambda x: x))
                    df[feature] = df[feature] * feature_weights.get(feature, 1)
            all_fundamentals.append(df)
    return pd.concat(all_fundamentals, ignore_index=True)

def load_time_series_data(time_series_dir):
    """Load and preprocess time series data for all tickers."""
    all_time_series = []
    for file in os.listdir(time_series_dir):
        if file.endswith(".csv"):
            print("File: ", file)
            ticker = file.split('_')[0]
            df = pd.read_csv(os.path.join(time_series_dir, file))
            df = df[['Date', 'Open', 'Close', 'Label']].copy()
            df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.split()[0]))
            df.set_index('Date',drop=True,inplace=True)
            df['Ticker'] = ticker
            df[['Open', 'Close']] = scaler.fit_transform(df[['Open', 'Close']])
            all_time_series.append(df)
    return pd.concat(all_time_series, ignore_index=True)

# Load and Preprocess Data
scaler = MinMaxScaler()

# Load data for all tickers
fundamentals_df = load_fundamental_data(fundamentals_dir)
time_series_df = load_time_series_data(time_series_dir)

print(f"Loaded fund series data shape: {fundamentals_df.shape}")
print(f"Sample fund series data:\n{fundamentals_df.head()}")

print(f"Loaded time series data shape: {time_series_df.shape}")
print(f"Sample time series data:\n{time_series_df.head()}")

def prepare_fundamental_data(df):
    """Prepare fundamental data for training."""
    X_fund = df[fundamental_features].values
    y_fund = df['label'].str.lower().map(label_mapping).values
    return X_fund, y_fund
def prepare_time_series_data(df, lookback=30):
    """Prepare time series data for training."""
    grouped_data = df.groupby('Ticker')
    all_X, all_y = [], []
    for ticker, group in grouped_data:
        group = group.dropna(subset=["Label"])
        group["Label"] = group["Label"].str.lower().map(label_mapping)
        if len(group) <= lookback:
            print(f"Not enough data for ticker {ticker} and lookback {lookback}.")
            continue
        X = np.array([group[time_series_features].iloc[i:i+lookback].values
                      for i in range(len(group) - lookback)])
        y = group["Label"].iloc[lookback:].values
        all_X.append(X)
        all_y.append(y)
    return np.vstack(all_X), np.concatenate(all_y)


def train_test_split(X, y, train_ratio=0.8):
    """Split data into train and test sets."""
    split_index = int(len(X) * train_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

label_mapping = {
    'nan' : -1,
    'strong buy': 0,
    'strongbuy': 0,
    'buy': 1,
    'hold': 2,
    'sell': 3,
    'strong sell': 4,
    'strongsell': 4
}

# Prepare Fundamental Data
X_fund, y_fund = prepare_fundamental_data(fundamentals_df)
print("Fundamental: ")
print(X_fund)
print(y_fund)

# Prepare Time Series Data
lookback = 30
X_ts, y_ts = prepare_time_series_data(time_series_df, lookback)
print("Time series: ")
print(X_ts)
print(y_ts)

# Split Data into Train and Test Sets


X_train_fund, X_test_fund, y_train_fund, y_test_fund = train_test_split(X_fund, y_fund, train_ratio=0.8)
X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(X_ts, y_ts, train_ratio=0.8)


y_train_ts = to_categorical(y_train_ts, num_classes=num_classes)
y_test_ts = to_categorical(y_test_ts, num_classes=num_classes)


print(f"y_train_ts shape: {y_train_ts.shape}")  # Should be (samples, num_classes)
print(f"y_test_ts shape: {y_test_ts.shape}")

# Verify Data Shapes
print(f"X_train_fund: {X_train_fund.shape}, X_test_fund: {X_test_fund.shape}")
print(f"X_train_ts: {X_train_ts.shape}, X_test_ts: {X_test_ts.shape}")
print(f"y_train: {y_train_ts.shape}, y_test: {y_test_ts.shape}")

l2_reg = regularizers.l2(0.01)

input_fund = Input(shape=(len(fundamental_features),), name='Fundamental_Input')
x_fund = Dense(64, activation='relu', kernel_regularizer=l2_reg)(input_fund)
x_fund = Dropout(0.2)(x_fund)
encoded_fund = Dense(32, activation='relu', name='Fundamental_Encoder')(x_fund)
x_decode = Dense(64, activation='relu')(encoded_fund)
output_fund = Dense(len(fundamental_features), activation='linear', name='Fundamental_Output')(x_decode)

# Train Autoencoder
autoencoder = Model(inputs=input_fund, outputs=output_fund, name='Fundamental_Autoencoder')
autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
autoencoder.fit(X_train_fund, X_train_fund, epochs=50, batch_size=32,
                validation_data=(X_test_fund, X_test_fund), verbose=1)

# Extract Pre-trained Encoder
fundamental_encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('Fundamental_Encoder').output)

# Encode Fundamental Features Once (Generalized Representation)
encoded_fund_combined = fundamental_encoder.predict(X_fund)
encoded_fund_mean = np.mean(encoded_fund_combined, axis=0)

input_ts = Input(shape=(lookback, len(time_series_features)), name='Time_Series_Input')
x_ts = LSTM(128, return_sequences=False, name='LSTM_Layer', kernel_regularizer=l2_reg)(input_ts)
x_ts = Dropout(0.2, name='Time_Series_Dropout')(x_ts)

# Combine Encoded Fundamental Features with Time Series Output
encoded_fund_general = Input(shape=(32,), name='General_Fundamental_Input')  # Generalized Encoder Input
combined = Concatenate(name='Combined_Features')([x_ts, encoded_fund_general])
x_combined = Dense(64, activation='relu', kernel_regularizer=l2_reg, name='Dense_Combined_64')(combined)
x_combined = Dropout(0.2, name='Combined_Dropout')(x_combined)
output = Dense(num_classes, activation='softmax', name='Final_Output')(x_combined)

# === Step 5: Build and Train the Hybrid Model === #
combined_model = Model(inputs=[input_ts, encoded_fund_general], outputs=output, name='Hybrid_Model')
combined_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# Duplicate Encoded Fundamental Features for Each Time Series Sample
X_train_fund_general = np.tile(encoded_fund_mean, (X_train_ts.shape[0], 1))
X_test_fund_general = np.tile(encoded_fund_mean, (X_test_ts.shape[0], 1))

# Train the Combined Model
history = combined_model.fit(
    [X_train_ts, X_train_fund_general], y_train_ts,
    epochs=50, batch_size=32,
    validation_data=([X_test_ts, X_test_fund_general], y_test_ts),
    verbose=1
)

class_names = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']

# Predictions
y_pred = combined_model.predict([X_test_ts, X_test_fund_general])  # Predict using test data
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert softmax output to class indices
y_test_classes = np.argmax(y_test_ts, axis=1)  # Convert one-hot encoded labels back to class indices

# Classification Report
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot Accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()