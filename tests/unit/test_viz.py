from conftest import ValueStorage
from sahi_tracking.utils.viz import SequenceVizualizer



def test_SequenceVizualizer_from_det_gt_pred():
    track_res = ValueStorage.state_persistance.state['tracking_results'][0]
    dataset_res = ValueStorage.state_persistance.load_data('datasets', track_res['dataset_hash'])
    detections_res = ValueStorage.state_persistance.load_data('predictions_results', track_res['predictions_hash'])

    viz = SequenceVizualizer.from_det_gt_pred('C0054_783015_783046', dataset_res, detections_res, track_res)
    viz.next_frame

def test_SequenceVizualizer_frame_gallery():
        track_res = ValueStorage.state_persistance.state['tracking_results'][0]
        dataset_res = ValueStorage.state_persistance.load_data('datasets', track_res['dataset_hash'])
        detections_res = ValueStorage.state_persistance.load_data('predictions_results', track_res['predictions_hash'])

        viz = SequenceVizualizer.from_det_gt_pred('C0054_783015_783046', dataset_res, detections_res, track_res)
        viz.create_frame_gallery(30, max_cols=6)
