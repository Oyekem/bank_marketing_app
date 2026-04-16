import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# LOAD MODEL
model = joblib.load ("xgboost_bank_pipeline.pkl")

# FEATURE FLAGS (GLOBAL CONTROL)
if "features" not in st.session_state:
    st.session_state.features = {
        "predict": True,
        "show_hidden_features": False,
        "debug_mode": False
    }


# SIDEBAR
st.sidebar.title("🏦 Bank App")

page = st.sidebar.radio("Go to", [
    "Home Page",
    "Prediction",
    "Data Visualization"
])


st.sidebar.markdown("### ⚙️ Feature Settings")

st.session_state.features["predict"] = st.sidebar.checkbox(
    "Enable Prediction",
    value=st.session_state.features["predict"]
)

st.session_state.features["show_hidden_features"] = st.sidebar.checkbox(
    "Show Advanced Features",
    value=st.session_state.features["show_hidden_features"]
)

st.session_state.features["debug_mode"] = st.sidebar.checkbox(
    "Debug Mode",
    value=st.session_state.features["debug_mode"]
)


# HOME PAGE
if page == "Home Page":

    st.title("🏦 Bank Marketing Prediction App")

    st.markdown("""
    ## Welcome to Tawakalt Prediction Application!

    This app predicts if a customer will subscribe to a term deposit.

    ### Project Overview
    - Built using Machine Learning (XGBoost)
    - Provides real-time predictions

    ### What You Can Do Here
    - 🔮 Predict customer subscription
    - 📊 Explore dataset insights

    ### Model Info
    - XGBoost Classifier
    - Binary Classification
    """)

    st.write("### Model Features")
    st.write(model.feature_names_in_)


# PREDICTION PAGE
elif page == "Prediction":

    st.title("🏦 Prediction Engine")

    expected_features = model.feature_names_in_

    # FEATURE DEFINITIONS
    categorical_options = {
        "job": ["admin.", "technician", "services", "management",
                "blue-collar", "self-employed", "entrepreneur"],
        "marital": ["single", "married", "divorced"],
        "education": ["primary", "secondary", "tertiary", "unknown"],
        "contact": ["cellular", "telephone", "unknown"],
        "month": ["jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec"],
        "poutcome": ["success", "failure", "other", "unknown"]
    }

    important_features = [
        "age", "job", "marital",
        "balance", "housing", "loan", "duration"
    ]

    st.write("### Customer Form")

    input_dict = {}

    # USER INPUT FIELDS
    for col in important_features:

        if col in ["age", "balance", "duration"]:
            input_dict[col] = st.number_input(col, value=0)

        elif col in ["housing", "loan"]:
            input_dict[col] = st.selectbox(col, ["yes", "no"])

        elif col == "job":
            input_dict[col] = st.selectbox("job", categorical_options["job"])

        elif col == "marital":
            input_dict[col] = st.selectbox("marital", categorical_options["marital"])


    # HIDDEN FEATURES (FIXED PROPERLY)
    if st.session_state.features["show_hidden_features"]:
        st.warning("⚠️ Advanced features enabled")

        hidden_defaults = {
            "campaign": st.number_input("campaign (hidden)", value=1),
            "pdays": st.number_input("pdays (hidden)", value=-1),
            "previous": st.number_input("previous (hidden)", value=0),
            "poutcome": st.selectbox(
                "poutcome (hidden)",
                ["success", "failure", "other", "unknown"]
            )
        }

    else:
        hidden_defaults = {
            "campaign": 1,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown"
        }

    # MERGE FEATURES
    input_dict.update(hidden_defaults)

    # FINAL DATAFRAME
    input_data = pd.DataFrame([input_dict])
    input_data = input_data.reindex(columns=expected_features, fill_value=0)

    st.write("### 🔍 Input Preview")
    st.dataframe(input_data)

    # DISABLE PREDICTION IF TURNED OFF
    if not st.session_state.features["predict"]:
        st.warning("🚫 Prediction feature is disabled")
        st.stop()

    # PREDICTION
    if st.button("Predict"):

        prob = model.predict_proba(input_data)[0][1]

        threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5)

        # classification
        prediction = 1 if prob >= threshold else 0

        if prediction == 1:
          st.success("✅ Customer WILL subscribe")
        else:
          st.error("❌ Customer will NOT subscribe")

        # interpretation layer
        if prob < 0.3:
          st.write("🔴 Very unlikely customer will subscribe")
        elif prob < 0.5:
          st.write("🟠 Unlikely customer will subscribe")
        elif prob < 0.7:
          st.write("🟡 Possible subscription")
        else:
          st.write("🟢 High chance of subscription")

        st.info(f"📊 Probability: {prob:.2%}")


    # DEBUG MODE
    if st.session_state.features["debug_mode"]:
        st.write("### 🐞 Debug Info")
        st.write("Input Types:", input_data.dtypes)
        st.write("Missing Values:", input_data.isnull().sum())


# DATA VISUALIZATION

# DATA VISUALIZATION
elif page == "Data Visualization":

    st.title("📊 Interactive Data Dashboard")

    import os


    # LOAD DATA (SAFE + CLEANED)
    try:
      df = pd.read_csv("cleaned_bank_data.csv")
      st.success("✅ Cleaned dataset loaded successfully")
    except:
      st.error("❌ Failed to load cleaned dataset, using fallback raw data")
      df = pd.read_csv("bank-full.csv")

    # CLEAN COLUMN NAMES
    df.columns = df.columns.str.strip()

    st.write("### Dataset Preview")

    df_display = df.copy()

    # 🔥 FIX STREAMLIT / PYARROW ISSUE
    df_display = df_display.convert_dtypes()

    for col in df_display.columns:
      if df_display[col].dtype == "object":
          try:
              df_display[col] = pd.to_numeric(df_display[col])
          except:
              df_display[col] = df_display[col].astype(str)

    st.dataframe(df_display.head(20))

    st.write("### Columns")
    st.write(df.columns)

    st.write("### Data Types")
    st.write(df.dtypes)


    # FILTER SYSTEM (POWER BI STYLE)
    st.sidebar.subheader("🔎 Filters")

    if "job" in df.columns:
        selected_job = st.sidebar.multiselect(
            "Job",
            df["job"].unique(),
            default=df["job"].unique()
        )
        df = df[df["job"].isin(selected_job)]

    if "marital" in df.columns:
        selected_marital = st.sidebar.multiselect(
            "Marital Status",
            df["marital"].unique(),
            default=df["marital"].unique()
        )
        df = df[df["marital"].isin(selected_marital)]

    if "age" in df.columns:
        min_age, max_age = int(df["age"].min()), int(df["age"].max())
        age_range = st.sidebar.slider(
            "Age Range",
            min_age,
            max_age,
            (min_age, max_age)
        )
        df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]


    # 1. AGE DISTRIBUTION
    if "age" in df.columns:
        st.subheader("📊 Age Distribution")

        fig = px.histogram(df, x="age", nbins=30)
        st.plotly_chart(fig)


    # 2. BALANCE DISTRIBUTION
    if "balance" in df.columns:
        st.subheader("💰 Balance Distribution")

        fig = px.histogram(df, x="balance", nbins=30)
        st.plotly_chart(fig)


    # 3. JOB DISTRIBUTION
    if "job" in df.columns:
        st.subheader("💼 Job Distribution")

        job_counts = df["job"].value_counts()

        fig = px.bar(
            x=job_counts.index,
            y=job_counts.values,
            labels={"x": "Job", "y": "Count"}
        )
        st.plotly_chart(fig)


    # 4. MARITAL STATUS
    if "marital" in df.columns:
        st.subheader("❤️ Marital Status")

        marital_counts = df["marital"].value_counts()

        fig = px.pie(
            names=marital_counts.index,
            values=marital_counts.values
        )
        st.plotly_chart(fig)


    # 5. CORRELATION HEATMAP
    st.subheader("🔥 Correlation Heatmap")

    df_clean = df.copy()
    df_clean = df_clean.convert_dtypes()

    for col in df_clean.columns:
      if df_clean[col].dtype == "object":
          try:
              df_clean[col] = pd.to_numeric(df_clean[col])
          except:
              df_clean[col] = df_clean[col].astype(str)

    numeric_df = df_clean.select_dtypes(include=["number"])
    corr = numeric_df.corr()
    
    if numeric_df.shape[1] > 1:

        corr = numeric_df.corr()

        fig, ax = plt.subplots()
        sns.heatmap(corr, cmap="coolwarm", ax=ax)

        st.pyplot(fig)

    else:
        st.warning("⚠️ Not enough numeric columns for correlation")


    # 6. SUBSCRIPTION RATE
    if "y" in df.columns:
        st.subheader("🎯 Subscription Outcome")

        y_counts = df["y"].value_counts()

        fig = px.bar(
            x=y_counts.index,
            y=y_counts.values,
            labels={"x": "Subscribed", "y": "Count"}
        )
        st.plotly_chart(fig)


    # 7. SUBSCRIPTION BY JOB
    if "y" in df.columns and "job" in df.columns:
        st.subheader("📌 Subscription by Job")

        fig = px.histogram(df, x="job", color="y", barmode="group")
        st.plotly_chart(fig)
