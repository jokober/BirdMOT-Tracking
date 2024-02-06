import streamlit as st

from sahi_tracking.evaluation.trackeval_utils import METRICS


def tracker_dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True, width='small')},
        use_container_width=True,
        column_order= ["Select", "TrackerName"] +  ["HOTA","DetA","AssA","DetRe","DetPr","AssRe","AssPr","LocA", "HOTA(0)"],
        disabled=df.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)

def one_sequence_dataframe_with_selections(df, default_tracking_results_hash, columns=[]):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    df_with_selections.loc[df_with_selections['tracking_results_hash'].isin(default_tracking_results_hash), "Select"] = True

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=False,
        column_config={"Select": st.column_config.CheckboxColumn(required=True, width='small')},
        use_container_width=True,
        column_order= ["Select", "TrackerName"]  + columns,
        disabled=df.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)

def dataframe_with_selections(df, columns=[]):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=False,
        column_config={"Select": st.column_config.CheckboxColumn(required=True, width='small')},
        use_container_width=True,
        column_order= ["Select", "tracker_name"]  + columns,
        disabled=df.columns,
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)