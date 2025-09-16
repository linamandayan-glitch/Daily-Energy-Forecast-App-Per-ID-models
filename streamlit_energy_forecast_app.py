import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

st.set_page_config(page_title="Daily Energy Forecast (Per-ID)", layout="wide")
st.title("Daily Energy Forecast — Per-ID Models")

# --- Sidebar: settings ---
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (id,date,value,...)", type=["csv"], key="upload_csv"
)
csv_url = st.sidebar.text_input("Or paste CSV URL (optional)", key="csv_url_input")

date_col = st.sidebar.text_input("Date column name", value="date", key="date_col")
value_col = st.sidebar.text_input("Value column name", value="value", key="value_col")
id_col = st.sidebar.text_input("ID column name", value="id", key="id_col")

lags = st.sidebar.number_input("Number of lag days to use", min_value=1, max_value=60, value=7, key="lags")
horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=1, max_value=30, value=7, key="horizon")
train_fraction = st.sidebar.slider("Training fraction (most recent used for test)", 0.5, 0.95, 0.9, key="train_frac")
random_state = st.sidebar.number_input("Random seed", value=42, key="random_seed")

train_button = st.sidebar.button("Train & Forecast", key="train_btn")

# --- Helper functions ---
def load_data(uploaded, url):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif url:
        df = pd.read_csv(url)
    else:
        return None
    return df

def prepare_series(df, date_col, value_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, value_col]].dropna()
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col).asfreq('D')
    df[value_col] = df[value_col].ffill().bfill()
    return df

def create_lag_features(series, nlags):
    df = pd.DataFrame({'y': series})
    for lag in range(1, nlags+1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['roll_mean_7_shift1'] = df['y'].rolling(window=min(7, nlags+1)).mean().shift(1)
    df['roll_std_7'] = df['y'].rolling(window=min(7, nlags+1)).std().shift(1).fillna(0)
    df = df.dropna()
    return df

def train_model(X_train, y_train, seed=42):
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=seed,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    return model

def recursive_forecast(model, last_known, nlags, horizon):
    preds = []
    buffer = list(last_known[-nlags:])
    for _ in range(horizon):
        features = buffer[-nlags:]
        roll_mean_7 = np.mean(buffer[-7:])
        roll_std_7 = np.std(buffer[-7:])
        features.extend([roll_mean_7, roll_std_7])
        X = np.array(features).reshape(1, -1)
        yhat = model.predict(X)[0]
        preds.append(yhat)
        buffer.append(yhat)
    return preds

# --- Main flow ---
data_df = load_data(uploaded_file, csv_url.strip())

if data_df is None:
    st.info(
        "Upload a CSV with at least columns (id, date, value). Example:\n\nid,date,value\nA,2020-01-01,12.3\nA,2020-01-02,13.1\nB,2020-01-01,7.2"
    )
    st.stop()

if not all(col in data_df.columns for col in [id_col, date_col, value_col]):
    st.error(f"CSV must contain columns: {id_col}, {date_col}, {value_col}")
    st.stop()

available_ids = data_df[id_col].unique().tolist()
selected_ids = st.sidebar.multiselect(
    "Select IDs to forecast", available_ids, default=available_ids[:1], key="select_ids"
)

if not selected_ids:
    st.warning("Please select at least one ID.")
    st.stop()

all_forecasts = []

if train_button:
    for current_id in selected_ids:
        st.subheader(f"Forecast for ID: {current_id}")
        df_id = data_df[data_df[id_col] == current_id]
        if df_id.empty:
            st.warning(f"No data for {current_id}")
            continue

        series_df = prepare_series(df_id, date_col, value_col)
        feat_df = create_lag_features(series_df[value_col], int(lags))
        split_idx = int(len(feat_df) * train_fraction)
        train = feat_df.iloc[:split_idx]
        test = feat_df.iloc[split_idx:]

        X_train = train.drop(columns=['y'])
        y_train = train['y']
        X_test = test.drop(columns=['y'])
        y_test = test['y']

        model = train_model(X_train, y_train, seed=int(random_state))
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # --- Display metrics ---
        st.write({
            'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
            'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        })

        # --- Plot historical + test predictions ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(series_df.index, series_df[value_col], label='Historical', color='blue')
        test_index = series_df.index[-len(y_test):]
        ax.plot(test_index, y_test, label='Test Actual', color='green', marker='o')
        ax.plot(test_index, y_pred_test, label='Test Predicted', color='orange', linestyle='--', marker='x')
        ax.set_title(f'ID: {current_id} — Historical & Test Predictions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        st.pyplot(fig)

        # --- Table with test actual vs predicted ---
        results_df = pd.DataFrame({
            'Date': test_index,
            'Actual': y_test.values,
            'Predicted': y_pred_test
        })
        st.dataframe(results_df)

        # --- Recursive forecast for future ---
        preds_future = recursive_forecast(model, series_df[value_col], int(lags), int(horizon))
        future_index = pd.date_range(start=series_df.index.max() + pd.Timedelta(days=1), periods=int(horizon), freq='D')
        forecast_df = pd.DataFrame({id_col: current_id, date_col: future_index, 'forecast': preds_future})
        all_forecasts.append(forecast_df)

        # --- Plot future forecast ---
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(series_df.index, series_df[value_col], label='Historical', color='blue')
        ax2.plot(future_index, preds_future, label='Future Forecast', linestyle='--', color='red')
        ax2.set_title(f'ID: {current_id} — Future Forecast')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value')
        ax2.legend()
        st.pyplot(fig2)

    if all_forecasts:
        combined_forecasts = pd.concat(all_forecasts)
        st.download_button(
            "Download all forecasts CSV",
            data=combined_forecasts.to_csv(index=False).encode('utf-8'),
            file_name="all_forecasts.csv",
            mime="text/csv"
        )

else:
    st.info("Adjust settings in the sidebar, select IDs, and click 'Train & Forecast' to get predictions.")

st.markdown("---")
st.caption("Notes: This app builds one model per selected ID, shows test predictions vs actuals, and visualises future forecasts.")
