import streamlit as st

from sahi_tracking.helper.config import get_list_of_sahi_models, get_list_of_sahi_prediction_params, \
    get_list_of_tracking_datasets, get_list_of_experiments, get_sahi_model_path_by_name, \
    get_sahi_prediction_params_path_by_name, get_tracking_dataset_path_by_name, get_tracking_experiment_path_by_name


def show_experiment_selection():
    selected_model = st.selectbox(
        "Select model",
        get_list_of_sahi_models(),
        index=None,
    )
    st.write('You selected:', selected_model)

    selected_prediction_params = st.selectbox(
        "Select prediction params",
        get_list_of_sahi_prediction_params(),
        index=None,
    )

    st.write('You selected:', selected_prediction_params)
    selected_dataset = st.selectbox(
        "Select dataset",
        get_list_of_tracking_datasets(),
        index=None,
    )

    st.write('You selected:', selected_dataset)
    selected_experiment = st.selectbox(
        "What experiments do you want see?",
        get_list_of_experiments() + ["All"],
        index=None,
        placeholder="Select experiments ...",
    )

    st.write('You selected:', selected_experiment)

    if selected_model and selected_prediction_params and selected_dataset and selected_experiment:
        st.session_state['model_path'] = get_sahi_model_path_by_name(selected_model)
        st.session_state['prediction_params_path'] = get_sahi_prediction_params_path_by_name(
            selected_prediction_params)
        st.session_state['dataset_path'] = get_tracking_dataset_path_by_name(selected_dataset)
        if selected_experiment != "All":
            st.session_state['experiment_path'] = get_tracking_experiment_path_by_name(selected_experiment)
        elif selected_experiment == "All":
            st.session_state['experiment_path'] = "All"

        st.success("Data loaded!")