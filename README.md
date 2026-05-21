# 📈 Stock Price Movement Prediction

A machine learning project that predicts short-term stock price movements (up or down) using technical analysis indicators.

> ⚠️ **Work in Progress** — Still under development, some bugs may exist.

---

## Models Used

- Logistic Regression (baseline)
- XGBoost (used in the app)
- LSTM

## Features

- Fetches stock data via `yfinance`
- Technical indicators: RSI, MACD, Bollinger Bands, EMA, Momentum
- Interactive Streamlit UI

## Installation & Usage

```bash
pip install streamlit yfinance xgboost scikit-learn joblib pandas numpy
```

1. Run `notebook.ipynb` to train and save the model
2. Then launch the app:

```bash
streamlit run app.py
```

---

> This project is for **educational purposes only** and does not constitute financial advice.

---

## Authors

- Erbol Zhanyshov
- Mert Laleli 
- İbrahim Kahraman 

---