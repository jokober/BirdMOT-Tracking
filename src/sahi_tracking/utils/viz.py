import math
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from distinctipy import distinctipy
from matplotlib import pyplot as plt

from sahi_tracking.helper.config import get_local_data_path
from sahi_tracking.helper.misc import image_file_formats


class SequenceVizualizer:
    def __init__(self, mot_img_path: Path, gt_path: Path, det_path: Path, track_res_path: Path,
                 show_detections: bool = True, show_gt: bool = True, show_track_res: bool = True,
                 show_all_tracks=False):
        self.show_detections = show_detections
        self.show_gt = show_gt
        self.show_track_res = show_track_res
        self.show_all_tracks = show_all_tracks

        self.image_files = {}
        for file_format in image_file_formats:
            for img_file in mot_img_path.glob(f'*{file_format}'):
                assert int(img_file.stem) not in self.image_files.keys(), f"Duplicate image file name: {img_file.stem}"
                self.image_files[int(img_file.stem)] = img_file

        self.current_frame = 0
        self.total_frames = len(self.image_files)

        self.det = np.loadtxt(det_path, dtype=np.float64, delimiter=',')
        self.gt = np.loadtxt(gt_path, dtype=np.float64, delimiter=',')
        self.track_res = np.loadtxt(track_res_path, dtype=np.float64, delimiter=',')

        if len(self.track_res) > 0:
            self.colors = distinctipy.get_colors(int(np.max(self.track_res[:, 1])) + 1)

    def next_frame(self):
        self.current_frame += 1
        self.img = self.load_image(self.current_frame)
        return self.annotate_frame(self.current_frame, self.img)

    def frame_by_number(self, frame_idx):
        img = self.load_image(frame_idx)
        img = self.annotate_frame(frame_idx, img)
        if self.show_all_tracks:
            img = self.annotate_compl_tracks_on_frame(img)
        return img

    def load_image(self, frame_number: int):
        img = cv.cvtColor(cv.imread(self.image_files[frame_number].as_posix()), cv.COLOR_BGR2RGB)
        if img is None:
            sys.exit("Could not read the image.")
        return img

    def annotate_compl_tracks_on_frame(self, img):
        if self.show_detections:
            for line in self.det:
                img = cv.rectangle(img, (int(line[2]), int(line[3])), (int(line[4]), int(line[5])), (0, 0, 255),
                                   3)  # top-left corner and bottom-right corner of rectangle
                img = cv.putText(
                    img=img,
                    text=f'{line[6]:.2f}',
                    org=(int(line[2]), int(line[5]) + 20),
                    fontFace=cv.FONT_HERSHEY_DUPLEX,
                    fontScale=1.0,
                    color=(0, 0, 255),
                    thickness=1
                )

        if self.show_track_res:
            for line in self.track_res:
                img = cv.rectangle(img, (int(line[2]), int(line[3])), (int(line[4]), int(line[5])),
                                   self.colors[int(line[1])] * np.array((255, 255, 255)),
                                   3)  # top-left corner and bottom-right corner of rectangle
        if self.show_gt:
            for line in self.gt:
                img = cv.rectangle(img, (int(line[2]), int(line[3])), (int(line[4]), int(line[5])), (0, 255, 0),
                                   3)  # top-left corner and bottom-right corner of rectangle
        # img = cv.resize(img, (1920, 1080), interpolation= cv.INTER_LINEAR)
        return img

    def annotate_frame(self, file_idx, img, draw_text=False):
        dt_in_frame = self.det[self.det[:, 0] == int(file_idx)]
        gt_in_frame = self.gt[self.gt[:, 0] == int(file_idx)]
        if len(self.track_res) > 0:
            track_res_in_frame = self.track_res[self.track_res[:, 0] == int(file_idx)]
        else:
            track_res_in_frame = []

        if self.show_detections:
            for line in dt_in_frame:
                img = cv.rectangle(img, (int(line[2]), int(line[3])), (int(line[4]), int(line[5])), (0, 0, 255),
                                   3)  # top-left corner and bottom-right corner of rectangle
                img = cv.putText(
                    img=img,
                    text=f'{line[6]:.2f}',
                    org=(int(line[2]), int(line[5]) + 20),
                    fontFace=cv.FONT_HERSHEY_DUPLEX,
                    fontScale=1.0,
                    color=(0, 0, 255),
                    thickness=1
                )

        if self.show_gt:
            for line in gt_in_frame:
                img = cv.rectangle(img, (int(line[2]), int(line[3])), (int(line[4]), int(line[5])), (0, 255, 0),
                                   3)  # top-left corner and bottom-right corner of rectangle

        if self.show_track_res:
            for line in track_res_in_frame:
                img = cv.rectangle(img, (int(line[2]), int(line[3])), (int(line[4]), int(line[5])),
                                   self.colors[int(line[1])] * np.array((255, 255, 255)),
                                   3)  # top-left corner and bottom-right corner of rectangle

        if draw_text:
            for i, (color, text) in enumerate(
                    (
                            ((0, 0, 255), "detection"),
                            ((0, 255, 0), "tracking ground truth"),
                            ((255, 0, 0), "tracking result")
                    ), 1):
                img = cv.putText(
                    img=img,
                    text=text,
                    org=(20, 25 * i),
                    fontFace=cv.FONT_HERSHEY_DUPLEX,
                    fontScale=1.0,
                    color=color,
                    thickness=1
                )

        # img = cv.resize(img, (1920, 1080), interpolation= cv.INTER_LINEAR)
        return img

    def create_video(self, fps=29, filestem: str = "bird_sequence") -> Path:
        video_path = get_local_data_path() / "work_dir" / f"{filestem}.webm"
        fourcc_mp4 = cv.VideoWriter_fourcc('V', 'P', '8', '0')
        out_mp4 = cv.VideoWriter(video_path.as_posix(), fourcc_mp4, fps, (3840, 2160))

        for frame_num in range(1, self.total_frames):
            frame = self.load_image(frame_num)
            ann_frame = self.annotate_frame(frame_num, frame)
            out_mp4.write(cv.cvtColor(ann_frame, cv.COLOR_RGB2BGR))
        out_mp4.release()
        return video_path

    def get_cropped_miniature_objects(self):
        instance_ids = np.unique(self.gt[:, 1])
        cropped_imgs_by_id = {}

        for instance_id in instance_ids:
            cropped_imgs_by_id[instance_id] = []
            img_annotations = self.gt[self.gt[:, 1] == instance_id]

            for img_ann in img_annotations:
                crop_size = max([img_ann[4] - img_ann[2], img_ann[5] - img_ann[3]])
                center = (img_ann[2] + (img_ann[4] - img_ann[2]) / 2, img_ann[3] + (img_ann[5] - img_ann[3]) / 2)
                img = self.load_image(img_ann[0])
                cropped_image = img[int(center[1] - crop_size / 2):int(center[1] + crop_size / 2),
                                int(center[0] - crop_size / 2): int(center[0] + crop_size / 2)]
                if cropped_image.shape[0] == crop_size and cropped_image.shape[0] == crop_size:
                    cropped_imgs_by_id[instance_id].append(cropped_image)

        return cropped_imgs_by_id

    def create_frame_gallery(self, max_rows: int, max_cols: int):
        instance_gallery_plots = []
        cropped_imgs_by_id = self.get_cropped_miniature_objects()
        for instance_id, imgs_list in cropped_imgs_by_id.items():
            frames_total = len(imgs_list)
            print(min(max_rows, int(math.ceil(frames_total / max_cols))))
            print(frames_total)
            print(max_cols)
            nrows = max(1, min(max_rows, int(math.ceil(frames_total / max_cols))))

            fig, axs = plt.subplots(nrows=nrows, ncols=max_cols, figsize=(max_cols, nrows),
                                    subplot_kw={'xticks': [], 'yticks': []})
            for idx, ax in enumerate(axs.flat):
                print(len(imgs_list))
                print(idx)
                if idx < len(imgs_list) - 1 and imgs_list[idx].shape[1] > 0:
                    print(imgs_list[idx].shape)
                    print(len(imgs_list[idx]))
                    print(type(imgs_list[idx]))
                    ax.imshow(imgs_list[idx], interpolation=None)
                else:
                    ax.axis('off')

                # ax.set_title(str(interp_method)
            plt.tight_layout()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
            plt.margins(0, 0)

            instance_gallery_plots.append(fig)
        return instance_gallery_plots

    @classmethod
    def from_det_gt_pred(cls, seq_name: str, dataset_res: dict, detections_res: dict, track_res: dict,
                         show_detections: bool = True, show_gt: bool = True, show_track_res: bool = True,
                         show_all_tracks: bool = False):
        mot_img_path = [seq for seq in dataset_res['dataset']['sequences'] if seq['name'] == seq_name][0][
                           'mot_path'] / 'img1'
        gt_path = [seq['mot_path'] for seq in dataset_res['dataset']['sequences'] if seq['name'] == seq_name][
                      0] / 'gt/gt.txt'
        det_path = detections_res['dir'] / seq_name / f"MOT/det/{seq_name}.txt"
        track_res_path = track_res['tracking_results']['result_data_path'] / f"{seq_name}.txt"

        return cls(mot_img_path, gt_path, det_path, track_res_path, show_detections, show_gt, show_track_res,
                   show_all_tracks)

    @classmethod
    def from_mot_folder(cls, path: Path):
        mot_img_path = [seq for seq in dataset_res['dataset']['sequences'] if seq['name'] == seq_name][0][
                           'mot_path'] / 'img1'
        gt_path = [seq['mot_path'] for seq in dataset_res['dataset']['sequences'] if seq['name'] == seq_name][
                      0] / 'gt/gt.txt'
        det_path = detections_res['dir'] / seq_name / f"MOT/det/{seq_name}.txt"
        track_res_path = track_res['tracking_results']['result_data_path'] / f"{seq_name}.txt"

        return cls(mot_img_path, gt_path, det_path, track_res_path, show_detections, show_gt, show_track_res,
                   show_all_tracks)
