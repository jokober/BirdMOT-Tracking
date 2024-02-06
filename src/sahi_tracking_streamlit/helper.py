import streamlit as st

default_keys = ['model_path', 'prediction_params_path', 'dataset_path', 'experiment_path']


def initialize_default_session_state():
    for key in default_keys:
        if key not in st.session_state:
            st.session_state[key] = None
