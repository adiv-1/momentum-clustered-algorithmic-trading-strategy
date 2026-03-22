import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def _create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def lstm_strategy(prices, seq_length=20, hidden_size=50, retrain_freq=21):
    signals = pd.DataFrame(index=prices.index)
    signals['price'] = prices['price']
    signals['signal'] = 0.0

    window_size = 252
    scaler = None
    model = None

    for i in range(seq_length + window_size, len(prices)):
        if (i - seq_length - window_size) % retrain_freq == 0 or model is None:
            train_prices = prices['price'].iloc[i - window_size - seq_length:i].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_train = scaler.fit_transform(train_prices.reshape(-1, 1))

            X, y = _create_sequences(scaled_train, seq_length)
            if len(X) < 20:
                continue

            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)

            model = LSTMModel(
                input_size=1, hidden_size=hidden_size,
                num_layers=2, output_size=1, dropout=0.2,
            )
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            model.train()
            for _ in range(50):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()

        if model is not None and scaler is not None:
            model.eval()
            with torch.no_grad():
                recent_prices = prices['price'].iloc[i - seq_length:i].values
                scaled_recent = scaler.transform(recent_prices.reshape(-1, 1))
                test_input = torch.FloatTensor(scaled_recent).unsqueeze(0)
                pred_scaled = model(test_input).item()
                prediction = scaler.inverse_transform([[pred_scaled]])[0][0]
                current_price = prices['price'].iloc[i]
                signals.loc[signals.index[i], 'signal'] = (
                    1.0 if prediction > current_price else 0.0
                )

    signals['positions'] = signals['signal'].diff()
    signals['returns'] = np.log(signals['price'] / signals['price'].shift(1))
    signals['strategy_returns'] = signals['signal'].shift(1) * signals['returns']
    return signals
