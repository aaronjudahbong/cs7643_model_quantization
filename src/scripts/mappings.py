# For our semantic segmentation task, we will align the taxonomy with the CityScapes evaluation taxonomy,
# with the exception of the 'license_plate' class which we will ignore during training and evaluation.
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

data_labels = {
    0: {"name": "unlabeled", "trainId": 255, "color": (0, 0, 0)},
    1: {"name": "ego_vehicle", "trainId": 255, "color": (0, 0, 0)},
    2: {"name": "rectification_border", "trainId": 255, "color": (0, 0, 0)},
    3: {"name": "out_of_roi", "trainId": 255, "color": (0, 0, 0)},
    4: {"name": "static", "trainId": 255, "color": (0, 0, 0)},
    5: {"name": "dynamic", "trainId": 255, "color": (111, 74, 0)},
    6: {"name": "ground", "trainId": 255, "color": (81, 0, 81)},
    7: {"name": "road", "trainId": 0, "color": (128, 64, 128)},
    8: {"name": "sidewalk", "trainId": 1, "color": (244, 35, 232)},
    9: {"name": "parking", "trainId": 255, "color": (250, 170, 160)},
    10: {"name": "rail_track", "trainId": 255, "color": (230, 150, 140)},
    11: {"name": "building", "trainId": 2, "color": (70, 70, 70)},
    12: {"name": "wall", "trainId": 3, "color": (102, 102, 156)},
    13: {"name": "fence", "trainId": 4, "color": (190, 153, 153)},
    14: {"name": "guard_rail", "trainId": 255, "color": (180, 165, 180)},
    15: {"name": "bridge", "trainId": 255, "color": (150, 100, 100)},
    16: {"name": "tunnel", "trainId": 255, "color": (150, 120, 90)},
    17: {"name": "pole", "trainId": 5, "color": (153, 153, 153)},
    18: {"name": "polegroup", "trainId": 255, "color": (153, 153, 153)},
    19: {"name": "traffic_light", "trainId": 6, "color": (250, 170, 30)},
    20: {"name": "traffic_sign", "trainId": 7, "color": (220, 220, 0)},
    21: {"name": "vegetation", "trainId": 8, "color": (107, 142, 35)},
    22: {"name": "terrain", "trainId": 9, "color": (152, 251, 152)},
    23: {"name": "sky", "trainId": 10, "color": (70, 130, 180)},
    24: {"name": "person", "trainId": 11, "color": (220, 20, 60)},
    25: {"name": "rider", "trainId": 12, "color": (255, 0, 0)},
    26: {"name": "car", "trainId": 13, "color": (0, 0, 142)},
    27: {"name": "truck", "trainId": 14, "color": (0, 0, 70)},
    28: {"name": "bus", "trainId": 15, "color": (0, 60, 100)},
    29: {"name": "caravan", "trainId": 255, "color": (0, 0, 90)},
    30: {"name": "trailer", "trainId": 255, "color": (0, 0, 110)},
    31: {"name": "train", "trainId": 16, "color": (0, 80, 100)},
    32: {"name": "motorcycle", "trainId": 17, "color": (0, 0, 230)},
    33: {"name": "bicycle", "trainId": 18, "color": (119, 11, 32)},
    -1: {"name": "license_plate", "trainId": 255, "color": (0, 0, 142)},
}

map_id_to_train_id = {k: v["trainId"] for k, v in data_labels.items()}
map_train_id_to_color = {v["trainId"]: v["color"] for k, v in data_labels.items() if v["trainId"] != 255}
map_train_id_to_name = {v["trainId"]: v["name"] for k, v in data_labels.items() if v["trainId"] != 255}