from operator import index
import streamlit as st
import sys
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os

if os.path.exists('./dataset.csv'):
    df_csv = pd.read_csv('dataset.csv', index_col=None)


with st.sidebar:
    st.image("images\Rapidailogo.png",width=300)
    st.title("RapidAI")
    choose = st.radio("Options", ['Upload', 'Data Profiling', 'Model Building', 'Download'])
    st.info("Accelerate Insights, Empower Innovation: Unleash the Power of RapidAI")


if choose == "Upload":
    st.title("Upload Your CSV Dataset")
    data_uploaded_csv = st.file_uploader("Upload Your CSV Dataset")
    if data_uploaded_csv:
        df_csv = pd.read_csv(data_uploaded_csv, index_col=None, encoding='ISO-8859-1')
        df_csv.to_csv('dataset.csv', index=None)
        st.dataframe(df_csv, width=900)


if choose == "Data Profiling":
    st.title("Rapid Exploratory Data Analysis")
    profile_df = df_csv.profile_report()
    st_profile_report(profile_df)
    profile_df = df_csv.profile_report()
    st_profile_report(profile_df)

if choose == "Model Building":

    option = st.selectbox("Select your problem type",
                          ("Select an option", "Regression", "Classification"))
    st.write('You selected:', option)

    if option == "Classification":
        from pycaret.classification import setup, compare_models, pull, save_model, load_model, plot_model,predict_model

        options_input = st.multiselect("Choose the input features", df_csv.columns)
        option_target = st.selectbox("Choose the target features", df_csv.columns)
        if st.button("Train Model"):
            X = df_csv[options_input]
            y = df_csv[option_target]
            setup(data=X, target=y)
            best_model_classification = compare_models()
            compare_df = pull(best_model_classification)
            st.dataframe(compare_df)
            st.info(best_model_classification)
            save_model(best_model_classification, 'best_model')
           
    if option == "Regression":
        from pycaret.regression import setup, compare_models, pull, save_model, load_model, plot_model, predict_model

        options_input = st.multiselect("Choose the input features", df_csv.columns)
        option_target = st.selectbox("Choose the target features", df_csv.columns)
        if st.button("Train Model"):
            X = df_csv[options_input]
            y = df_csv[option_target]
            setup(data=X, target=y)
            best_model_regression = compare_models()
            compare_df = pull(best_model_regression)
            st.dataframe(compare_df)
            st.info(best_model_regression)
            save_model(best_model_regression, 'best_model')
            
if choose == "Download":
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")

 
