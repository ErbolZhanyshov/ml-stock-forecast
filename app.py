import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import date

st.set_page_config(
    page_title="Hisse Senedi Tahmin Sistemi",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Hisse Senedi Yükseliş / Düşüş Tahmin Sistemi")
st.write("Bu uygulama teknik analiz göstergelerine göre hissenin kısa vadede yükselme olasılığını tahmin eder.")

FEATURE_COLUMNS = [
    'RSI',
    'MACD',
    'Volatility',
    'Return',
    'Smoothed_Return',
    'Upper_Band',
    'Lower_Band'
]

@st.cache_resource
def load_model_files():
    model = joblib.load("xgb_model.joblib")
    scaler = joblib.load("scaler_features.joblib")
    return model, scaler

@st.cache_data(ttl=3600)
def load_data(ticker, start_date):
    today = date.today().strftime("%Y-%m-%d")

    data = yf.download(
        ticker,
        start=start_date,
        end=today,
        progress=False,
        auto_adjust=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)
    return data

def create_features(df):
    df = df.copy()

    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA30'] = df['Close'].rolling(30).mean()

    rolling_std = df['Close'].rolling(20).std()

    df['Upper_Band'] = df['MA30'] + (rolling_std * 2)
    df['Lower_Band'] = df['MA30'] - (rolling_std * 2)

    df['EMA10'] = df['Close'].ewm(span=10).mean()
    df['EMA30'] = df['Close'].ewm(span=30).mean()

    df['Trend'] = (df['EMA10'] > df['EMA30']).astype(int)

    df['Volatility'] = df['Close'].rolling(20).std()

    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']

    delta = df['Close'].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()

    df['MACD'] = ema12 - ema26

    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['Volume_Change'] = df['Volume'].pct_change()

    df['Return'] = df['Close'].pct_change()
    df['Smoothed_Return'] = df['Return'].rolling(5).mean()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df

try:
    model, scaler = load_model_files()

    st.sidebar.header("⚙️ Ayarlar")

    ticker = st.sidebar.text_input(
        "Hisse kodu",
        value="TCS.NS"
    )

    start_date = st.sidebar.text_input(
        "Başlangıç tarihi",
        value="2010-01-01"
    )

    threshold = st.sidebar.slider(
        "Tahmin eşiği",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.01
    )

    if st.sidebar.button("Tahmin Yap"):
        with st.spinner("Veriler çekiliyor ve göstergeler hesaplanıyor..."):
            raw_data = load_data(ticker, start_date)

            if raw_data.empty:
                st.error("Veri çekilemedi. Hisse kodunu kontrol et.")
            else:
                df_features = create_features(raw_data)

                if df_features.empty:
                    st.error("Yeterli veri yok. Daha eski bir başlangıç tarihi gir.")
                else:
                    latest_row = df_features.iloc[[-1]]

                    X_latest = latest_row[FEATURE_COLUMNS]
                    X_scaled = scaler.transform(X_latest)

                    prob = model.predict_proba(X_scaled)[0][1]
                    pred = int(prob > threshold)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Son Kapanış", f"{latest_row['Close'].values[0]:.2f}")

                    with col2:
                        st.metric("Yükseliş Olasılığı", f"%{prob * 100:.2f}")

                    with col3:
                        if pred == 1:
                            st.success("Tahmin: YÜKSELİŞ 📈")
                        else:
                            st.warning("Tahmin: DÜŞÜŞ / ZAYIF SİNYAL 📉")

                    st.subheader("Kapanış Fiyat Grafiği")

                    chart_df = df_features[['Date', 'Close']].copy()
                    chart_df = chart_df.set_index('Date')

                    st.line_chart(chart_df)

                    st.subheader("Son Hesaplanan Teknik Göstergeler")
                    st.dataframe(latest_row[FEATURE_COLUMNS])

                    st.info(
                        "Not: Bu sistem yatırım tavsiyesi değildir. "
                        "Model geçmiş fiyat verileri ve teknik göstergeler üzerinden tahmin üretir."
                    )

except FileNotFoundError:
    st.error("Model dosyaları bulunamadı. Önce xgb_model.joblib ve scaler_features.joblib dosyalarını oluşturmalısın.")
except Exception as e:
    st.error(f"Hata oluştu: {e}")
