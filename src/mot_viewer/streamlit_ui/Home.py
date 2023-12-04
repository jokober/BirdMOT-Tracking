import os

import streamlit as st
import tkinter as tk
from tkinter import filedialog

from sahi_tracking.helper.config import get_local_data_path
from sahi_tracking.utils.viz import SequenceVizualizer


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def select_folder():
   root = tk.Tk()
   root.withdraw()
   folder_path = filedialog.askdirectory(master=root)
   root.destroy()
   return folder_path

selected_folder_path = st.session_state.get("folder_path", None)
folder_select_button = st.button("Select Folder")
if folder_select_button:
  selected_folder_path = select_folder()
  st.session_state.folder_path = selected_folder_path

  if selected_folder_path:
      st.write("Selected folder path:", selected_folder_path)
      st.write("Selected folder path:", selected_folder_path)

mot_path = get_local_data_path()
path_iterator = mot_path.iterdir()

show_detections = st.checkbox("Show Detections")
show_gt = st.checkbox("Show Ground Truth (Tracking)")
show_track_res = st.checkbox("Show Tracking Results")
show_all_tracks = st.checkbox("Annotate whole sequence on current frame")

viz_obj = SequenceVizualizer.from_det_gt_pred(df_seq_select['seq_name'].to_list()[0], dataset_res, detections_res,
                                              track_res, show_detections=show_detections, show_gt=show_gt,
                                              show_track_res=show_track_res, show_all_tracks=show_all_tracks)

# Fill layout
with st.container.container():

    for idx, plt in enumerate(viz_obj.create_frame_gallery(max_rows=2, max_cols=6)):
        with st.expander(f"{idx}"):
            st.pyplot(plt, use_container_width=False)

with st.expander("Video"):
    fps = st.number_input("FPS", value=29)
    video_path = viz_obj.create_video(fps)
    video_file = open(video_path.as_posix(), 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)
    st.download_button('Download Video', video_bytes, file_name=video_path.name)