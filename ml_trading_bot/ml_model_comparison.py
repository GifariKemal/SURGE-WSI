"""
ML Model Comparison for Trading
===============================
Compare scikit-learn, TensorFlow/Keras models for GBPUSD H1 prediction.
Compare with RSI v3.7 baseline.

Author: SURIOTA Team
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import sys

warnings.filterwarnings('ignore')

def log(msg):
    print(msg)
    sys.stdout.flush()

# Check available libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    log("XGBoost not available")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
except ImportError:
    TF_AVAILABLE = False
    log("TensorFlow not available")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log("PyTorch not available")

# MT5 Connection
MT5_LOGIN = 61045904
MT5_PASSWORD = "iy#K5L7sF"
MT5_SERVER = "FinexBisnisSolusi-Demo"
MT5_PATH = r"C:\Program Files\Finex Bisnis Solusi MT5 Terminal\terminal64.exe"

def connect_mt5():
    if not mt5.initialize(path=MT5_PATH):
        return False
    if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
        return False
    return True

def get_data(symbol, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def create_features(df, lookback=20):
    """Create features for ML models"""
    features = pd.DataFrame(index=df.index)

    # Price-based features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Moving averages
    for period in [5, 10, 20, 50]:
        features[f'sma_{period}'] = df['close'].rolling(period).mean()
        features[f'price_vs_sma_{period}'] = (df['close'] - features[f'sma_{period}']) / features[f'sma_{period}']

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    features['atr'] = tr.rolling(14).mean()
    features['atr_pct'] = features['atr'] / df['close']

    # Volatility
    features['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

    # Momentum
    for period in [5, 10, 20]:
        features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_hist'] = features['macd'] - features['macd_signal']

    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    features['bb_upper'] = sma20 + 2 * std20
    features['bb_lower'] = sma20 - 2 * std20
    features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)

    # Stochastic
    low14 = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    features['stoch_k'] = 100 * (df['close'] - low14) / (high14 - low14 + 1e-10)
    features['stoch_d'] = features['stoch_k'].rolling(3).mean()

    # Time features
    features['hour'] = df.index.hour
    features['dayofweek'] = df.index.dayofweek
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['day_sin'] = np.sin(2 * np.pi * features['dayofweek'] / 7)
    features['day_cos'] = np.cos(2 * np.pi * features['dayofweek'] / 7)

    # Lagged features
    for lag in [1, 2, 3, 5]:
        features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        features[f'rsi_lag_{lag}'] = features['rsi'].shift(lag)

    # Drop non-numeric columns
    features = features.select_dtypes(include=[np.number])

    return features

def create_labels(df, lookahead=12, threshold=0.003):
    """Create labels: 1=Buy, -1=Sell, 0=Hold"""
    future_returns = df['close'].shift(-lookahead) / df['close'] - 1
    labels = np.zeros(len(df))
    labels[future_returns > threshold] = 1   # Buy
    labels[future_returns < -threshold] = -1  # Sell
    return labels

def run_rsi_baseline(df):
    """RSI v3.7 baseline for comparison"""
    # RSI calculation
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # ATR
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                              abs(df['low'] - df['close'].shift(1))))
    atr = tr.rolling(14).mean()

    def atr_pct_func(x):
        if len(x) == 0:
            return 50.0
        current = x[-1]
        count_below = (x[:-1] < current).sum()
        return (count_below / (len(x) - 1)) * 100 if len(x) > 1 else 50.0

    atr_pct = atr.rolling(100).apply(atr_pct_func, raw=True)

    balance = 10000.0
    wins = losses = 0
    position = None
    monthly_pnl = {}

    RSI_OS, RSI_OB = 42, 58
    SL_MULT = 1.5
    TP_LOW, TP_MED, TP_HIGH = 2.4, 3.0, 3.6
    MAX_HOLDING = 46

    df_copy = df.copy()
    df_copy['rsi'] = rsi
    df_copy['atr'] = atr
    df_copy['atr_pct'] = atr_pct
    df_copy['hour'] = df_copy.index.hour
    df_copy['weekday'] = df_copy.index.dayofweek
    df_copy = df_copy.ffill().fillna(0)

    for i in range(200, len(df_copy)):
        row = df_copy.iloc[i]
        current_time = df_copy.index[i]
        month_str = current_time.strftime('%Y-%m')

        if row['weekday'] >= 5:
            continue

        if position:
            exit_reason = None
            pnl = 0

            if (i - position['entry_idx']) >= MAX_HOLDING:
                pnl = (row['close'] - position['entry']) * position['size'] if position['dir'] == 1 else (position['entry'] - row['close']) * position['size']
                exit_reason = 'TIMEOUT'
            else:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        exit_reason = 'SL'
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        exit_reason = 'TP'
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        exit_reason = 'SL'
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        exit_reason = 'TP'

            if exit_reason:
                balance += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                if month_str not in monthly_pnl:
                    monthly_pnl[month_str] = 0
                monthly_pnl[month_str] += pnl
                position = None

        if not position:
            hour = row['hour']
            if hour < 7 or hour >= 22 or hour == 12:
                continue

            atr_pct_val = row['atr_pct']
            if atr_pct_val < 20 or atr_pct_val > 80:
                continue

            rsi_val = row['rsi']
            signal = 1 if rsi_val < RSI_OS else (-1 if rsi_val > RSI_OB else 0)

            if signal:
                entry = row['close']
                atr_val = row['atr'] if row['atr'] > 0 else entry * 0.002
                base_tp = TP_LOW if atr_pct_val < 40 else (TP_HIGH if atr_pct_val > 60 else TP_MED)
                tp_mult = base_tp + 0.35 if 12 <= hour < 16 else base_tp
                sl = entry - atr_val * SL_MULT if signal == 1 else entry + atr_val * SL_MULT
                tp = entry + atr_val * tp_mult if signal == 1 else entry - atr_val * tp_mult
                risk = balance * 0.01
                size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}

    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades else 0
    total_return = (balance - 10000) / 10000 * 100
    profitable_months = sum(1 for m in monthly_pnl.values() if m > 0)
    losing_months = sum(1 for m in monthly_pnl.values() if m < 0)

    return {
        'name': 'RSI v3.7 Baseline',
        'total_return': total_return,
        'trades': total_trades,
        'win_rate': win_rate,
        'profitable_months': profitable_months,
        'losing_months': losing_months
    }

def train_sklearn_models(X_train, y_train, X_test, y_test):
    """Train and evaluate scikit-learn models"""
    results = []

    # Random Forest
    log("  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results.append({
        'name': 'Random Forest',
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, rf_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, rf_pred, average='weighted', zero_division=0),
        'model': rf
    })

    # Gradient Boosting
    log("  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    results.append({
        'name': 'Gradient Boosting',
        'accuracy': accuracy_score(y_test, gb_pred),
        'precision': precision_score(y_test, gb_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, gb_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, gb_pred, average='weighted', zero_division=0),
        'model': gb
    })

    # XGBoost
    if XGB_AVAILABLE:
        log("  Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train + 1)  # XGBoost needs 0-indexed labels
        xgb_pred = xgb_model.predict(X_test) - 1
        results.append({
            'name': 'XGBoost',
            'accuracy': accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, xgb_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, xgb_pred, average='weighted', zero_division=0),
            'model': xgb_model
        })

    return results

def train_tensorflow_models(X_train, y_train, X_test, y_test):
    """Train and evaluate TensorFlow/Keras models"""
    if not TF_AVAILABLE:
        return []

    results = []

    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train + 1, num_classes=3)
    y_test_cat = keras.utils.to_categorical(y_test + 1, num_classes=3)

    # Dense Neural Network
    log("  Training Dense Neural Network...")
    dnn = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    dnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
    dnn.fit(X_train, y_train_cat, epochs=50, batch_size=64, validation_split=0.2,
            callbacks=[early_stop], verbose=0)
    dnn_pred = np.argmax(dnn.predict(X_test, verbose=0), axis=1) - 1
    results.append({
        'name': 'Dense NN (TensorFlow)',
        'accuracy': accuracy_score(y_test, dnn_pred),
        'precision': precision_score(y_test, dnn_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, dnn_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, dnn_pred, average='weighted', zero_division=0),
        'model': dnn
    })

    # LSTM
    log("  Training LSTM...")
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    lstm = Sequential([
        LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lstm.fit(X_train_lstm, y_train_cat, epochs=30, batch_size=64, validation_split=0.2,
             callbacks=[early_stop], verbose=0)
    lstm_pred = np.argmax(lstm.predict(X_test_lstm, verbose=0), axis=1) - 1
    results.append({
        'name': 'LSTM (TensorFlow)',
        'accuracy': accuracy_score(y_test, lstm_pred),
        'precision': precision_score(y_test, lstm_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, lstm_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, lstm_pred, average='weighted', zero_division=0),
        'model': lstm
    })

    return results

def train_pytorch_model(X_train, y_train, X_test, y_test):
    """Train and evaluate PyTorch model"""
    if not TORCH_AVAILABLE:
        return []

    log("  Training PyTorch Neural Network...")

    class TradingNN(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 3)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = self.dropout(self.relu(self.bn1(self.fc1(x))))
            x = self.dropout(self.relu(self.bn2(self.fc2(x))))
            x = self.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    # Prepare data
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train + 1)  # 0, 1, 2
    X_test_t = torch.FloatTensor(X_test)

    model = TradingNN(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        pytorch_pred = predicted.numpy() - 1

    return [{
        'name': 'Neural Network (PyTorch)',
        'accuracy': accuracy_score(y_test, pytorch_pred),
        'precision': precision_score(y_test, pytorch_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, pytorch_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, pytorch_pred, average='weighted', zero_division=0),
        'model': model
    }]

def backtest_ml_model(df, model, scaler, features, lookahead=12):
    """Backtest ML model with same rules as RSI v3.7"""
    balance = 10000.0
    wins = losses = 0
    position = None
    monthly_pnl = {}

    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    df_copy['weekday'] = df_copy.index.dayofweek

    # ATR for SL/TP
    tr = np.maximum(df_copy['high'] - df_copy['low'],
                    np.maximum(abs(df_copy['high'] - df_copy['close'].shift(1)),
                              abs(df_copy['low'] - df_copy['close'].shift(1))))
    df_copy['atr'] = tr.rolling(14).mean()

    feature_cols = [c for c in features.columns if c in df_copy.columns or c in features.columns]

    SL_MULT = 1.5
    TP_MULT = 2.5
    MAX_HOLDING = 46

    for i in range(200, len(df_copy) - lookahead):
        row = df_copy.iloc[i]
        current_time = df_copy.index[i]
        month_str = current_time.strftime('%Y-%m')

        if row['weekday'] >= 5:
            continue

        if position:
            exit_reason = None
            pnl = 0

            if (i - position['entry_idx']) >= MAX_HOLDING:
                pnl = (row['close'] - position['entry']) * position['size'] if position['dir'] == 1 else (position['entry'] - row['close']) * position['size']
                exit_reason = 'TIMEOUT'
            else:
                if position['dir'] == 1:
                    if row['low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) * position['size']
                        exit_reason = 'SL'
                    elif row['high'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) * position['size']
                        exit_reason = 'TP'
                else:
                    if row['high'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) * position['size']
                        exit_reason = 'SL'
                    elif row['low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) * position['size']
                        exit_reason = 'TP'

            if exit_reason:
                balance += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                if month_str not in monthly_pnl:
                    monthly_pnl[month_str] = 0
                monthly_pnl[month_str] += pnl
                position = None

        if not position:
            hour = row['hour']
            if hour < 7 or hour >= 22:
                continue

            # Get features for this row
            try:
                feat_row = features.iloc[i:i+1].values
                if np.isnan(feat_row).any():
                    continue
                feat_scaled = scaler.transform(feat_row)

                # Get prediction
                if hasattr(model, 'predict_proba'):
                    pred = model.predict(feat_scaled)[0]
                elif hasattr(model, 'predict'):
                    if 'keras' in str(type(model)):
                        pred = np.argmax(model.predict(feat_scaled, verbose=0), axis=1)[0] - 1
                    else:
                        pred = model.predict(feat_scaled)[0]
                else:
                    continue

                signal = int(pred)
                if signal == 0:
                    continue

                entry = row['close']
                atr = row['atr'] if row['atr'] > 0 else entry * 0.002
                sl = entry - atr * SL_MULT if signal == 1 else entry + atr * SL_MULT
                tp = entry + atr * TP_MULT if signal == 1 else entry - atr * TP_MULT
                risk = balance * 0.01
                size = min(risk / abs(entry - sl), 100000) if abs(entry - sl) > 0 else 0
                position = {'dir': signal, 'entry': entry, 'sl': sl, 'tp': tp, 'size': size, 'entry_idx': i}

            except Exception as e:
                continue

    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades else 0
    total_return = (balance - 10000) / 10000 * 100
    profitable_months = sum(1 for m in monthly_pnl.values() if m > 0)
    losing_months = sum(1 for m in monthly_pnl.values() if m < 0)

    return {
        'total_return': total_return,
        'trades': total_trades,
        'win_rate': win_rate,
        'profitable_months': profitable_months,
        'losing_months': losing_months
    }

def main():
    log("=" * 70)
    log("ML MODEL COMPARISON FOR TRADING")
    log("scikit-learn | TensorFlow | PyTorch")
    log("=" * 70)

    log(f"\nLibraries available:")
    log(f"  scikit-learn: Yes")
    log(f"  XGBoost: {XGB_AVAILABLE}")
    log(f"  TensorFlow: {TF_AVAILABLE}")
    log(f"  PyTorch: {TORCH_AVAILABLE}")

    if not connect_mt5():
        log("MT5 connection failed")
        return

    try:
        log("\nFetching GBPUSD H1 data...")
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc)
        df = get_data("GBPUSD", start_date, end_date)

        if df is None:
            log("Failed to get data")
            return

        log(f"Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")

        # Create features and labels
        log("\nCreating features...")
        features = create_features(df)
        labels = create_labels(df, lookahead=12, threshold=0.003)

        # Remove NaN
        valid_idx = ~(features.isna().any(axis=1) | np.isnan(labels))
        features = features[valid_idx]
        labels = labels[valid_idx]
        df_valid = df[valid_idx]

        log(f"Features: {features.shape[1]} columns, {len(features)} rows")
        log(f"Label distribution: Buy={sum(labels==1)}, Sell={sum(labels==-1)}, Hold={sum(labels==0)}")

        # Train/test split (time-based)
        split_idx = int(len(features) * 0.7)
        X_train = features.iloc[:split_idx].values
        y_train = labels[:split_idx]
        X_test = features.iloc[split_idx:].values
        y_test = labels[split_idx:]

        log(f"\nTrain: {len(X_train)} samples, Test: {len(X_test)} samples")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models
        all_results = []

        log("\n" + "=" * 70)
        log("TRAINING MODELS")
        log("=" * 70)

        # Scikit-learn models
        log("\n[1] Scikit-learn Models:")
        sklearn_results = train_sklearn_models(X_train_scaled, y_train, X_test_scaled, y_test)
        all_results.extend(sklearn_results)

        # TensorFlow models
        if TF_AVAILABLE:
            log("\n[2] TensorFlow Models:")
            tf_results = train_tensorflow_models(X_train_scaled, y_train, X_test_scaled, y_test)
            all_results.extend(tf_results)

        # PyTorch models
        if TORCH_AVAILABLE:
            log("\n[3] PyTorch Models:")
            pytorch_results = train_pytorch_model(X_train_scaled, y_train, X_test_scaled, y_test)
            all_results.extend(pytorch_results)

        # Classification Results
        log("\n" + "=" * 70)
        log("CLASSIFICATION RESULTS (Test Set)")
        log("=" * 70)
        log(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        log("-" * 65)

        for r in all_results:
            log(f"{r['name']:<25} {r['accuracy']:.4f}     {r['precision']:.4f}     {r['recall']:.4f}     {r['f1']:.4f}")

        # Backtest comparison
        log("\n" + "=" * 70)
        log("BACKTEST COMPARISON (Trading Performance)")
        log("=" * 70)

        # RSI v3.7 baseline
        log("\nRunning RSI v3.7 baseline...")
        rsi_result = run_rsi_baseline(df_valid.iloc[split_idx:])

        log("\nRunning ML model backtests...")
        backtest_results = [rsi_result]

        for r in all_results:
            log(f"  Backtesting {r['name']}...")
            bt_result = backtest_ml_model(
                df_valid.iloc[split_idx:],
                r['model'],
                scaler,
                features.iloc[split_idx:],
                lookahead=12
            )
            bt_result['name'] = r['name']
            backtest_results.append(bt_result)

        # Print backtest results
        log("\n" + "-" * 70)
        log(f"{'Model':<25} {'Return':<10} {'Trades':<8} {'WinRate':<10} {'ProfitMo':<10} {'LossMo':<8}")
        log("-" * 70)

        for r in backtest_results:
            log(f"{r['name']:<25} {r['total_return']:>7.1f}%   {r['trades']:<8} {r['win_rate']:>7.1f}%   "
                f"{r['profitable_months']:<10} {r['losing_months']:<8}")

        # Best model
        best = max(backtest_results, key=lambda x: x['total_return'])
        log("\n" + "=" * 70)
        log(f"BEST MODEL: {best['name']}")
        log(f"Return: {best['total_return']:.2f}%")
        log(f"Trades: {best['trades']}")
        log(f"Win Rate: {best['win_rate']:.1f}%")
        log("=" * 70)

    finally:
        mt5.shutdown()
        log("\nDone!")

if __name__ == "__main__":
    main()
