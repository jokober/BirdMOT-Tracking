from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

from sahi_tracking.evaluation.trackeval_plotting import load_trackeval_evaluation_data, plot_compare_trackers, \
    create_comparison_plot
from sahi_tracking.experiments_framework.DataStatePersistance import DataStatePersistance
from sahi_tracking.experiments_framework.utils import load_config_files
from sahi_tracking.evaluation.trackeval_utils import trackeval_to_pandas, evaluation_results_to_pandas, METRICS, \
    get_linspaced_metric_names
from sahi_tracking.utils.viz import SequenceVizualizer

from sahi_tracking_streamlit.experiment_selection import show_experiment_selection
from sahi_tracking_streamlit.helper import initialize_default_session_state
from sahi_tracking_streamlit.load_data import load_data, get_tracking_experiment_conf
from sahi_tracking_streamlit.ui_components.dataframes import dataframe_with_selections, \
    tracker_dataframe_with_selections, one_sequence_dataframe_with_selections

initialize_default_session_state()
def next(): st.session_state.counter += 1
def prev(): st.session_state.counter -= 1
def first(): st.session_state.counter = 1
def last(): st.session_state.counter -= 1
if 'counter' not in st.session_state: st.session_state.counter = 1

st.set_page_config(layout="wide")

st.markdown("# 0 Experiment Selection")
show_experiment_selection()

if st.session_state['dataset_path'] and st.session_state['experiment_path'] and st.session_state['prediction_params_path']:
    if st.session_state['experiment_path'] != "All":
        tracking_dataset_dict, sahi_predictions_params_dict, tracking_experiment_dict = load_config_files(
            st.session_state['dataset_path'], Path(st.session_state['experiment_path']),
            Path(st.session_state['prediction_params_path']))
        state_persistance, dataset, predictions_result, evaluation_results_list = load_data(tracking_dataset_dict, sahi_predictions_params_dict, tracking_experiment_dict)
    if st.session_state['experiment_path'] == "All":
        state_persistance = DataStatePersistance()
        evaluation_results_list = state_persistance.state['evaluation_results']

    st.markdown("# 1 Tracker Overview and Selection")
    for eval in evaluation_results_list:
        # Add fps from tracking resul
        tracking_result = state_persistance.load_data('tracking_results', eval['tracking_results_hash'])['tracking_results']
        print("eval")
        print(type(eval))
        print(type(tracking_result))
        if 'fps' in tracking_result:
            eval['fps'] = tracking_result['fps']
        else:
            eval['fps'] = None

    df = evaluation_results_to_pandas(evaluation_results_list)
    df_select = tracker_dataframe_with_selections(df.loc["COMBINED_SEQ"])
    filtered_tracking_hash = df_select['tracking_results_hash']

    st.markdown("# 2 Tracker Results")
    df_select = df_select.sort_values(by=['HOTA'],ignore_index=True, ascending=False)
    df_select.index += 1
    st.dataframe(df_select, column_order=['TrackerName','HOTA', 'DetA', 'AssA', 'LocA', 'fps' ])

    selected_eval_results =  [state_persistance.load_data('evaluation_results', eval_hash) for eval_hash in df_select['eval_results_hash'].to_list()]
    selected_trackeval_data = load_trackeval_evaluation_data(selected_eval_results, "pedestrian")

    with st.expander("AssA vs DetA (HOTA)"):
        st.markdown(
        """
        **AssA:** Average alignment between matched trajectories, averaged over all detections.  
        **DetA:** Percentage of aligning detections.
        """)
        st.pyplot(create_comparison_plot(selected_trackeval_data, "/media/data/BirdMOT/local_data_tracking/", *['AssA', 'DetA', 'HOTA', 'HOTA', 'geometric_mean']), use_container_width=False)

    with st.expander("AssPr vs AssRe (AssA)"):
        st.markdown(
        """
        **AssRe:** Measures how well trackers can avoid splitting the same object into multiple shorter tracks  
        **AssPr:** Measures how well tracks can avoid merging multiple objects together into a single track
        """)
        st.pyplot(create_comparison_plot(selected_trackeval_data, "/media/data/BirdMOT/local_data_tracking/", *['AssPr', 'AssRe', 'HOTA', 'AssA', 'jaccard']), use_container_width=False)

    with st.expander("DetPr vs DetRe (DetA)"):
        st.markdown(
        """
        **DetPr:** Measures how well a tracker manages to not predict extra detections that aren’t there.  
        **DetRe:** Measures how well a tracker finds all the ground-truth detections
        """)
        st.pyplot(create_comparison_plot(selected_trackeval_data, "/media/data/BirdMOT/local_data_tracking/", *['DetPr', 'DetRe', 'HOTA', 'DetA', 'jaccard']), use_container_width=False)

    with st.expander("HOTA(0) vs LocA(0) (HOTALocA(0))"):
        st.markdown(
        """
        **HOTA(0):** HOTA at the single lowest alpha threshold, so to not include the influence of localization accuracy)   
        **LocA(0):** LocA at the same threshold 
        """)
        st.pyplot(create_comparison_plot(selected_trackeval_data, "/media/data/BirdMOT/local_data_tracking/", *['HOTA(0)', 'LocA(0)', 'HOTA', 'HOTALocA(0)', 'multiplication']), use_container_width=False)

    with st.expander("HOTA vs LocA"):
        st.markdown(
        """
        **:**  
        **:** 
        """)
        st.pyplot(create_comparison_plot(selected_trackeval_data, "/media/data/BirdMOT/local_data_tracking/", *['HOTA', 'LocA', 'HOTA', None, None]), use_container_width=False)


    #for plt in plot_compare_trackers(trackeval_data, "pedestrian", output_folder = "/media/data/BirdMOT/local_data_tracking/", plots_list= None):
    #    st.pyplot(plt)

    #detections_res = state_persistance.load_data('predictions_results', track_res['predictions_hash'])
    #df = trackeval_to_pandas(evaluation_results_list[0]['evaluation_results'][0]['MotChallenge2DBox']['default_tracker'])
    st.markdown('# 3 Sequence Selection')

    # Optional Column Select
    unique_tracking_hashes = pd.unique(df['tracking_results_hash'])
    tracking_param_set = set()
    for tracking_hash in unique_tracking_hashes:
        tracking_param_set.update(get_tracking_experiment_conf(tracking_hash)['tracking_experiment']['tracker_config'].keys())
    tracking_param_options = st.multiselect(
        'Select tracker parameters to add',
       tracking_param_set)
    if len(tracking_param_options) > 0:
        for param in tracking_param_options:
            for tracking_hash in unique_tracking_hashes:
                df.loc[df['tracking_results_hash'] == tracking_hash, param] = get_tracking_experiment_conf(tracking_hash)['tracking_experiment']['tracker_config'].get(param)

    # Filter Column Selection
    filter_param_set = set()
    filter_param_dict_by_hash = {}
    for tracking_hash in unique_tracking_hashes:
        if 'filter' in get_tracking_experiment_conf(tracking_hash)['tracking_experiment']:
            filter_conf_df = pd.json_normalize(get_tracking_experiment_conf(tracking_hash)['tracking_experiment']['filter'], sep='_')
            filter_conf_dict = filter_conf_df.to_dict(orient='records')[0]
            filter_param_dict_by_hash[tracking_hash] = filter_conf_dict
            filter_param_set.update(filter_conf_dict.keys())
    filter_param_options = st.multiselect(
        'Select filter parameters to add',
       filter_param_set)
    if len(filter_param_options) > 0:
        for param in filter_param_options:
            for tracking_hash in unique_tracking_hashes:
                if tracking_hash in filter_param_dict_by_hash:
                    df.loc[df['tracking_results_hash'] == tracking_hash, param] = filter_param_dict_by_hash[tracking_hash][param]
                    tracking_hash
                    filter_param_dict_by_hash
                    filter_param_dict_by_hash.get(tracking_hash, "blah")
                    filter_param_dict_by_hash[tracking_hash][param]

    # Metrics Column Selection
    metrics_options = st.multiselect(
        'Select metrics to show as columns',
       ['DetA___5', 'AssA___5', 'DetRe___5', 'DetPr___5', 'AssRe___5', 'AssPr___5'] + METRICS)


    prefix_filter = st.text_input('Contains-Filter', '')

    df = df[df.index.str.contains(prefix_filter)]
    df_seq_select = dataframe_with_selections(df[
                                                  df['tracking_results_hash'].isin(filtered_tracking_hash.to_list()) &
                                                  (df['seq_name'] != 'COMBINED_SEQ')
                                              ],
                                              columns=metrics_options + tracking_param_options  + filter_param_options)

    st.markdown("# 4 Sequences")
    filtered_seq_tracking_hash = df_seq_select['tracking_results_hash'].to_list()
    filtered_seq_seq_name = df_seq_select['seq_name'].to_list()

    comparison_sequence_tracker = one_sequence_dataframe_with_selections(df[
                                                  df['seq_name'].isin(filtered_seq_seq_name)
                                              ],
                                             default_tracking_results_hash = filtered_seq_tracking_hash,
                                                                         columns =metrics_options + tracking_param_options  + filter_param_options
                                             )
    filtered_seq_tracking_hash = comparison_sequence_tracker['tracking_results_hash'].to_list()

    # Image
    if len(filtered_seq_tracking_hash) == 0:
        st.warning('No sequence selected. Select sequence in order to show images', icon="⚠️")
    else:
        track_res = get_tracking_experiment_conf(filtered_seq_tracking_hash[0])
        dataset_res = state_persistance.load_data('datasets', track_res['dataset_hash'])
        detections_res = state_persistance.load_data('predictions_results', track_res['predictions_hash'])

        show_detections = st.checkbox("Show Detections")
        show_gt = st.checkbox("Show Ground Truth (Tracking)")
        show_track_res = st.checkbox("Show Tracking Results")
        show_all_tracks = st.checkbox("Annotate whole sequence on current frame")

        viz_obj = SequenceVizualizer.from_det_gt_pred(df_seq_select['seq_name'].to_list()[0], dataset_res, detections_res,
                                                      track_res, show_detections= show_detections, show_gt= show_gt,
                                                      show_track_res= show_track_res, show_all_tracks = show_all_tracks)
        img = viz_obj.next_frame()


        container = st.empty()
        cols = st.columns(5)
        with cols[4]: st.button("Next ➡️", on_click=next, use_container_width=True)
        with cols[3]: st.button("Last", on_click=last, use_container_width=True)
        with cols[2]: st.text(str(st.session_state.counter) + '/' + str(viz_obj.total_frames))
        with cols[1]: st.button("First", on_click=first, use_container_width=True)
        with cols[0]: st.button("⬅️ Previous", on_click=prev, use_container_width=True)


        # Fill layout
        with container.container():
            st.image(viz_obj.frame_by_number(st.session_state.counter))

            for idx, plt in enumerate(viz_obj.create_frame_gallery(max_rows=4, max_cols=8)):
                plt.savefig(f"gallery{df_seq_select['seq_name'].to_list()[0]}_{str(idx)}.png")
                with st.expander(f"{idx}"):
                    st.pyplot(plt, use_container_width=False)

    with st.expander("Video"):
        fps = st.number_input("FPS", value=29)
        video_path = viz_obj.create_video(fps, f"video_{df_seq_select['seq_name'].to_list()[0]}")
        video_file = open(video_path.as_posix(), 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)
        st.download_button('Download Video', video_bytes, file_name=video_path.name)