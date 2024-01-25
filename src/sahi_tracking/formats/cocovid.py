from typing import List


def filter_sequences(coco_dict: dict, sequences_to_keep: List[str]) -> dict:
    """
    The filter_sequences function filters the coco_dict dictionary to only contain the sequences in the sequences_to_keep list.
    The function returns a dictionary with the filtered sequences.

    """
    videos =[it for it in coco_dict['videos'] if it['name'] in sequences_to_keep]
    video_ids = [it['id'] for it in videos]

    images=[it for it in coco_dict['images'] if int(it['video_id']) in video_ids]
    image_ids= [it['id'] for it in images]

    annotations=[it for it in coco_dict['annotations'] if int(it['image_id']) in image_ids]

    return {
        'videos' : videos,
        'images' : images,
        'annotations' : annotations,
        'categories' : coco_dict['categories']
    }
