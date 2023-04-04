import streamlit as st
import plotly.express as px 
import os
import pandas as pd
import ydata_profiling as yd_profiling
from streamlit_pandas_profiling import st_profile_report


with st.sidebar: 
    st.title("RapidAI")
    choice=st.radio("Navigation",["Upload","Data Profiling",st.selectbox('RapidML', ['Classification', 'Regression']),"Download"])
    st.info("Accelerate Insights, Empower Innovation: Unleash the Power of RapidAI")

if os.path.exists("dataset.csv"): 
    df = pd.read_csv('dataset.csv', index_col=None)


if choice == "Upload": 
    st.title("Upload")
    file = st.file_uploader("Upload Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv("dataset.csv", index=None)
        st.dataframe(df)

if choice == "Data Profiling": 
    st.title("Profile")
    profile_df = df.profile_report()
    st_profile_report(profile_df)


if choice == "Classification":
    target = st.selectbox("Choose the Target for Classification", df.columns)
    if st.button("Run Classification"):
        from pycaret.classification import setup, compare_models, pull, save_model

        setup(df, target=target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        save_model(best_model, 'best_model')
        st.dataframe(compare_df)

if choice == "Regression":
    target = st.selectbox("Choose the Target for Regression", df.columns)
    if st.button("Run Regression"):
        from pycaret.regression import setup, compare_models, pull, save_model

        setup(df, target=target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        save_model(best_model, 'best_model')
        st.dataframe(compare_df)
    
if choice == "Download":
    with open("best_model.pkl", 'rb') as f: 
        st.download_button("Download Model", f, "best_model_test.pkl")


    
