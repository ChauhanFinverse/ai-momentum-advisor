import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.fetch_data import fetch_stock_data
from src.feature_engineer import create_features, create_labels

nse500 = pd.read_csv('data/nse500_list.csv')['Symbol'].tolist()

def retrain_model():
    data = fetch_stock_data([symbol + ".NS" for symbol in nse500], "2022-01-01", None)
    features = create_features(data)
    labels = create_labels(data)
    aligned = features.align(labels, join='inner', axis=0)
    X, y = aligned[0].dropna(), aligned[1].dropna()
    y = y.loc[X.index].mean(axis=1).round()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "models/rf_model.pkl")