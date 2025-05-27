import pandas as pd
import joblib
from src.fetch_data import fetch_stock_data
from src.feature_engineer import create_features

nse500 = pd.read_csv('data/nse500_list.csv')['Symbol'].tolist()

def get_today_predictions():
    data = fetch_stock_data([s + ".NS" for s in nse500], "2023-10-01", None)
    features = create_features(data)
    today = features.iloc[-1:]

    model = joblib.load("models/rf_model.pkl")
    prob = model.predict_proba(today)[0][1] if hasattr(model, 'predict_proba') else model.predict(today)[0]
    return pd.DataFrame({"Stock": [nse500[0]], "Momentum Score": [round(prob, 3)]})