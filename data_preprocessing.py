import os
import pandas as pd
import csv
import time
import numpy as np
import yfinance as yfin
from sklearn.preprocessing import MinMaxScaler
from datetime import date
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

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

DATA_DIR = "C:/Local Desktop/Year 6/Machine Learning/Project/"
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
        time.sleep(5)
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
with ThreadPoolExecutor(max_workers=1) as executor:
    executor.map(lambda stock: fetch_fundamental_data(stock), S_AND_P_500)
    #executor.map(lambda stock: fetch_stock_data(stock, start_time, end_time), S_AND_P_500)