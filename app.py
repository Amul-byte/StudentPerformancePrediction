import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📊 Student Performance Analysis & Prediction")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("./data/StudentsPerformance.csv")
    df["Average Score"] = ((df["math score"] + df["reading score"] + df["writing score"]) / 3).round(2)
    return df

df = load_data()

# Encode categorical columns
def encode_data(df):
    le = LabelEncoder()
    for col in ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]:
        df[col] = le.fit_transform(df[col])
    return df

df_encoded = encode_data(df.copy())

# Sidebar
st.sidebar.header("Navigation")
view = st.sidebar.radio("Choose view", ["Dataset", "Visualizations", "Model"])

# Dataset
if view == "Dataset":
    st.subheader("📄 Raw Dataset (Top 20 Rows)")
    st.dataframe(df.head(20))

# Visualizations
elif view == "Visualizations":
    st.subheader("📊 Visualizations")

    # Countplot for gender vs race
    st.markdown("### Gender vs Race/Ethnicity Count")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="gender", hue="race/ethnicity", data=df, ax=ax1)
    st.pyplot(fig1)

    # Pie chart - test preparation course
    st.markdown("### Test Preparation Course Completion")
    pie_labels = ["Not Completed", "Completed"]
    pie_colors = ["red", "green"]
    fig2, ax2 = plt.subplots()
    ax2.pie(df["test preparation course"].value_counts(), labels=pie_labels, colors=pie_colors, autopct="%1.1f%%")
    st.pyplot(fig2)

    # Barplot - test prep vs average score
    st.markdown("### Test Prep vs Average Score")
    fig3, ax3 = plt.subplots()
    sns.barplot(x="test preparation course", y="Average Score", data=df_encoded, ax=ax3)
    st.pyplot(fig3)

    # Barplot - lunch vs average score
    st.markdown("### Lunch Type vs Average Score")
    fig4, ax4 = plt.subplots()
    sns.barplot(x="lunch", y="Average Score", data=df_encoded, ax=ax4, palette=["red", "green"])
    st.pyplot(fig4)

    # Barplot - parental education vs average score
    st.markdown("### Parental Level of Education vs Average Score")
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    sns.barplot(x="parental level of education", y="Average Score", data=df_encoded, ax=ax5, palette="inferno")
    st.pyplot(fig5)

    # Heatmap
    st.markdown("### Correlation Heatmap")
    fig6, ax6 = plt.subplots(figsize=(10, 5))
    sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", ax=ax6)
    st.pyplot(fig6)

# Model
elif view == "Model":
    st.subheader("🤖 Linear Regression Model")

    # Prepare data
    X = df_encoded.drop(["Average Score", "math score", "reading score", "writing score"], axis=1)
    y = df_encoded["Average Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Show predictions vs actual
    result_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    st.dataframe(result_df.head(10))

    # Mean error
    error = abs(y_test - y_pred).mean()
    st.markdown(f"### 📉 Mean Absolute Error: `{round(error, 2)}`")

    # Plot actual vs predicted
    st.markdown("### Actual vs Predicted Scores")
    fig7, ax7 = plt.subplots()
  # Scatter plot
    fig7, ax7 = plt.subplots()

# Plot actual values (as blue dots on diagonal)
    ax7.scatter(y_test, y_test, color="green", label="Actual", alpha=0.6)

    # Plot predicted values (green dots)
    ax7.scatter(y_test, y_pred, color="blue", label="Predicted", alpha=0.6)

    # Set labels and title
    ax7.set_xlabel("Actual")
    ax7.set_ylabel("Predicted")
    ax7.set_title("Actual vs Predicted Scores")

    # Add legend
    ax7.legend()

    st.pyplot(fig7)

