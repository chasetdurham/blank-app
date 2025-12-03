import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os

st.title("Snowfall Prediction Model")
st.write("This app predicts future snowfall based on previous days' totals.")

# -------------------------------
# Load or create snowfall history
# -------------------------------

HISTORY_FILE = "snow_history.csv"

# Create a blank file if it doesn't exist
if not os.path.exists(HISTORY_FILE):
    df = pd.DataFrame({"day": [], "snowfall": []})
    df.to_csv(HISTORY_FILE, index=False)

df = pd.read_csv(HISTORY_FILE)

# -------------------------------
# Enter today's snowfall
# -------------------------------
st.subheader("Enter Today's Snowfall (in inches)")
today_snow = st.number_input("Snowfall:", min_value=0.0, max_value=200.0, step=0.1)

if st.button("Add Snowfall Entry"):
    new_row = {"day": len(df) + 1, "snowfall": today_snow}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)
    st.success("Snowfall added!")

# -------------------------------
# Show snowfall history
# -------------------------------
st.subheader("Previous Days Snowfall")
if len(df) > 0:
    st.dataframe(df)
else:
    st.write("No snowfall entries yet.")

# -------------------------------
# Prediction Settings
# -------------------------------
st.subheader("Prediction Settings")

if len(df) >= 2:
    lookback = st.slider(
        "How many past days to use in the model?",
        min_value=1,
        max_value=min(30, len(df) - 1),
        value=3
    )
else:
    lookback = 1

# -------------------------------
# Train model and predict tomorrow
# -------------------------------
if len(df) > lookback:
    snow_values = df["snowfall"].values

    # Create training sequences
    X = []
    y = []

    for i in range(len(snow_values) - lookback):
        X.append(snow_values[i:i + lookback])
        y.append(snow_values[i + lookback])

    X = np.array(X)
    y = np.array(y)

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Predict tomorrow
    last_sequence = np.array([snow_values[-lookback:]])
    prediction = model.predict(last_sequence)[0]

    st.subheader("Predicted Snowfall for Tomorrow")
    st.metric("Prediction", f"{prediction:.2f} inches")

    # -------------------------------
    # Plot snowfall history
    # -------------------------------
    st.subheader("Snowfall History Chart")

    fig, ax = plt.subplots()
    ax.plot(df["day"], df["snowfall"], marker="o")
    ax.set_xlabel("Day")
    ax.set_ylabel("Snowfall (in)")
    ax.set_title("Snowfall Over Time")
    st.pyplot(fig)

else:
    st.info(f"Need at least {lookback + 1} days of data to build the model.")
