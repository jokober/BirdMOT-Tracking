from typing import List

import cv2
import numpy as np
from sahi.prediction import PredictionResult

from sahi_tracking.trackers.iou_tracker.util import iou
from sahi_tracking.trackers.iou_tracker.viou_tracker import associate
from sahi_tracking.trackers.iou_tracker.vis_tracker import VisTracker


class VIoUTracker:
    def __init__(self, img_path, sigma_l=0.3, sigma_h=0.7, sigma_iou=0.3, t_min=3, ttl=10, tracker_type="KCF",
                 keep_upper_height_ratio=0.0):
        if tracker_type == 'NONE':
            assert ttl == 1, "ttl should not be larger than 1 if no visual tracker is selected"
        self.img_path = img_path
        self.sigma_l = sigma_l
        self.sigma_h = sigma_h
        self.sigma_iou = sigma_iou
        self.t_min = t_min
        self.ttl = ttl
        self.tracker_type = tracker_type
        self.keep_upper_height_ratio = keep_upper_height_ratio

        self.matrix_predictions = np.empty((0, 10), dtype=float)
        self.frame_number = 0

        self.name = self.create_tracker_name(tracker_type)

        self.tracks_extendable = []
        self.tracks_active = []
        self.tracks_finished = []
        self.frame_buffer = []

    def offline_tracking(self, object_prediction_list: PredictionResult):
        raise NotImplementedError("The offline tracking is not implemented.")

    def update_online(self, object_prediction_list: List[PredictionResult]):
        self.frame_number += 1

        detections_frame = [{
            'score': pred_res.score.value,
            'bbox': [round(v) for v in pred_res.bbox.to_xyxy()],
            'class': pred_res.category.id
        } for pred_res in object_prediction_list]

        # load frame and put into buffer
        frame_path = next(self.img_path.glob(f"{self.frame_number:06d}.*"))
        frame = cv2.imread(frame_path.as_posix())
        assert frame is not None, "could not read '{}'".format(frame_path)
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.ttl + 1:
            self.frame_buffer.pop(0)

        # apply low threshold to detections
        dets = [det for det in detections_frame if det['score'] >= self.sigma_l]

        track_ids, det_ids = associate(self.tracks_active, dets, self.sigma_iou)
        updated_tracks = []
        for track_id, det_id in zip(track_ids, det_ids):
            self.tracks_active[track_id]['bboxes'].append(dets[det_id]['bbox'])
            self.tracks_active[track_id]['max_score'] = max(self.tracks_active[track_id]['max_score'],
                                                            dets[det_id]['score'])
            self.tracks_active[track_id]['classes'].append(dets[det_id]['class'])
            self.tracks_active[track_id]['det_counter'] += 1

            if self.tracks_active[track_id]['ttl'] != self.ttl:
                # reset visual tracker if active
                self.tracks_active[track_id]['ttl'] = self.ttl
                self.tracks_active[track_id]['visual_tracker'] = None

            updated_tracks.append(self.tracks_active[track_id])

        tracks_not_updated = [self.tracks_active[idx] for idx in
                              set(range(len(self.tracks_active))).difference(set(track_ids))]

        for track in tracks_not_updated:
            if track['ttl'] > 0:
                if track['ttl'] == self.ttl:
                    # init visual tracker
                    track['visual_tracker'] = VisTracker(self.tracker_type, track['bboxes'][-1], self.frame_buffer[-2],
                                                         self.keep_upper_height_ratio)
                # viou forward update
                ok, bbox = track['visual_tracker'].update(frame)

                if not ok:
                    # visual update failed, track can still be extended
                    self.tracks_extendable.append(track)
                    continue

                track['ttl'] -= 1
                track['bboxes'].append(bbox)
                updated_tracks.append(track)
            else:
                self.tracks_extendable.append(track)

        # update the list of extendable tracks. tracks that are too old are moved to the finished_tracks. this should
        # not be necessary but may improve the performance for large numbers of tracks (eg. for mot19)
        self.tracks_extendable_updated = []
        for track in self.tracks_extendable:
            if track['start_frame'] + len(track['bboxes']) + self.ttl - track['ttl'] >= self.frame_number:
                self.tracks_extendable_updated.append(track)
            elif track['max_score'] >= self.sigma_h and track['det_counter'] >= self.t_min:
                self.tracks_finished.append(track)
        self.tracks_extendable = self.tracks_extendable_updated

        new_dets = [dets[idx] for idx in set(range(len(dets))).difference(set(det_ids))]
        dets_for_new = []

        for det in new_dets:
            finished = False
            # go backwards and track visually
            boxes = []
            vis_tracker = VisTracker(self.tracker_type, det['bbox'], frame, self.keep_upper_height_ratio)
            # i = cv2.rectangle(frame, (int(det['bbox'][0]), int(det['bbox'][1])),
            #                   (int(det['bbox'][2]), int(det['bbox'][3])), color=(0, 255, 0), thickness=2)
            #
            # i = cv2.resize(i, (1920, 1080))
            # cv2.imshow("asd", i)
            # k = cv2.waitKey(0)

            for f in reversed(self.frame_buffer[:-1]):
                ok, bbox = vis_tracker.update(f)
                if not ok:
                    # can not go further back as the visual tracker failed
                    break
                boxes.append(bbox)

                # sorting is not really necessary but helps to avoid different behaviour for different orderings
                # preferring longer tracks for extension seems intuitive, LAP solving might be better
                for track in sorted(self.tracks_extendable, key=lambda x: len(x['bboxes']), reverse=True):

                    offset = track['start_frame'] + len(track['bboxes']) + len(boxes) - self.frame_number
                    # association not optimal (LAP solving might be better)
                    # association is performed at the same frame, not adjacent ones
                    if 1 <= offset <= self.ttl - track['ttl'] and iou(track['bboxes'][-offset], bbox) >= self.sigma_iou:
                        if offset > 1:
                            # remove existing visually tracked boxes behind the matching frame
                            track['bboxes'] = track['bboxes'][:-offset + 1]
                        track['bboxes'] += list(reversed(boxes))[1:]
                        track['bboxes'].append(det['bbox'])
                        track['max_score'] = max(track['max_score'], det['score'])
                        track['classes'].append(det['class'])
                        track['ttl'] = self.ttl
                        track['visual_tracker'] = None

                        self.tracks_extendable.remove(track)
                        if track in self.tracks_finished:
                            del self.tracks_finished[self.tracks_finished.index(track)]
                        updated_tracks.append(track)

                        finished = True
                        break
                if finished:
                    break
            if not finished:
                dets_for_new.append(det)

        # create new tracks
        new_tracks = [
            {'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': self.frame_number, 'ttl': self.ttl,
             'classes': [det['class']], 'det_counter': 1, 'visual_tracker': None} for det in dets_for_new]
        self.tracks_active = []
        for track in updated_tracks + new_tracks:
            if track['ttl'] == 0:
                self.tracks_extendable.append(track)
            else:
                self.tracks_active.append(track)

        # ToDo: Currently there are not returned tracks even though it is the online update function ...

    def collect_results(self):
        # finish all remaining active and extendable tracks
        self.tracks_finished = self.tracks_finished + \
                               [track for track in self.tracks_active + self.tracks_extendable
                                if track['max_score'] >= self.sigma_h and track['det_counter'] >= self.t_min]

        # remove last visually tracked frames and compute the track classes
        for track in self.tracks_finished:
            if self.ttl != track['ttl']:
                track['bboxes'] = track['bboxes'][:-(self.ttl - track['ttl'])]
            track['class'] = max(set(track['classes']), key=track['classes'].count)

            del track['visual_tracker']

        id_ = 1
        for track in self.tracks_finished:
            for i, bbox in enumerate(track['bboxes']):
                new_row = [
                    track['start_frame'] + i,
                    id_,
                    bbox[0] + 1,
                    bbox[1] + 1,
                    bbox[2],
                    bbox[3],
                    -1,  # track['max_score']
                    -1,
                    -1,
                    -1,
                ]
                self.matrix_predictions = np.vstack((self.matrix_predictions, new_row))
            id_ += 1

    def get_mot_list(self):
        return self.matrix_predictions

    def create_tracker_name(self, tracker_type):
        name = f"V-IoU ({tracker_type})"
        return name
