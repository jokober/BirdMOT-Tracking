import configparser
import csv
from pathlib import Path

from sahi_tracking.formats.mot_format import create_mot_folder_structure


def cocovid2mot(cocovid_dict: dict, out_dir: Path, image_path: Path):
    """
    Converts a COCOVID annotation dict to a MOT annotation dict.
    """
    # ToDo: handle width and height and frame rate
    sequences = []
    for vid in cocovid_dict['videos']:
        mot_path, mot_img_path, det_path, gt_path = create_mot_folder_structure(vid['name'], out_dir)

        images_in_vid = [it for it in cocovid_dict['images'] if int(it['video_id']) == vid['id']]
        seqLength = len(images_in_vid)

        # Create seqinfo.ini file
        config = configparser.ConfigParser()
        seqinfo_ini = Path(f"{mot_path}/seqinfo.ini")  # Path of your .ini file
        config["Sequence"] = {
            'name': f"MOT-{vid['name']}",
            'imDir': 'img1',
            'frameRate': '29.97',
            'seqLength': f'{seqLength}',
            'imWidth': '3840',
            'imHeight': '2160',
        }
        config.write(seqinfo_ini.open("w"), space_around_delimiters=False)

        # Create image symlinks
        for idx, img in enumerate(images_in_vid, 1):
            symlink_path = mot_img_path / f"{idx:06d}{Path(img['file_name']).suffix}"
            assert not symlink_path.exists(), f"""Symlink path {symlink_path} already exists. This might be due
            to a failed previous attempt to create the dataset. Try to remove the assoziated dataset folder and try again.
            """
            symlink_path.symlink_to(image_path / img['file_name'])

        # Write ground truth annotation data
        with open(det_path, 'w') as det, open(gt_path, 'w') as gt:
            # create the csv writer
            gt_writer = csv.writer(gt)

            for img in images_in_vid:
                annotations_in_img = [it for it in cocovid_dict['annotations'] if
                                      int(it['image_id']) == img['id']]

                for ann in annotations_in_img:
                    # write a row to the csv file
                    bb_xywh = ann['bbox']

                    print(img)
                    gt_writer.writerow([img['frame_id'] + 1, ann['instance_id'],  # gt_bb.class_id,
                                        bb_xywh[0],
                                        bb_xywh[1],
                                        bb_xywh[2] + bb_xywh[0],
                                        bb_xywh[3] + bb_xywh[1],
                                        1, 1,  # gt_bb.class_id,
                                        -1])
        sequences.append({
            'name': vid['name'],
            'mot_path': mot_path
        })
    return sequences
