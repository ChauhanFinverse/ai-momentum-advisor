import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_backtest_plot():
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=120)
    strategy = (1 + np.random.normal(0.001, 0.01, len(dates))).cumprod()
    nifty = (1 + np.random.normal(0.0005, 0.009, len(dates))).cumprod()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dates, strategy, label="AI Strategy")
    ax.plot(dates, nifty, label="NIFTY 50")
    ax.set_title("Backtest Comparison")
    ax.legend()
    return fig