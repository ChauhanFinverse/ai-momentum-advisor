import streamlit as st
from src.predict import get_today_predictions
from src.backtest import get_backtest_plot

st.set_page_config(page_title="AI Momentum Advisor", layout="centered")
st.title("📈 AI-Based Momentum Stock Advisor")

st.subheader("Today's Top Momentum Picks")
top_stocks = get_today_predictions()
st.table(top_stocks)

st.subheader("Backtest Performance (Last 6 Months)")
fig = get_backtest_plot()
st.pyplot(fig)

if st.button("Retrain Model"):
    from src.train_model import retrain_model
    retrain_model()
    st.success("Model retrained successfully!")