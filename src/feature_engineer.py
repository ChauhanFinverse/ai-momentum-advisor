import pandas as pd

def create_features(prices):
    returns = prices.pct_change()
    features = pd.DataFrame(index=returns.index)
    features['5d_return'] = returns.rolling(5).sum()
    features['10d_return'] = returns.rolling(10).sum()
    features['20d_return'] = returns.rolling(20).sum()
    features['volatility'] = returns.rolling(10).std()
    return features.dropna()

def create_labels(prices):
    fwd_return = prices.shift(-10).pct_change(10)
    label = (fwd_return > 0.05).astype(int)
    return label