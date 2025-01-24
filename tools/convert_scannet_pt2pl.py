import os
import os.path as osp

import pickle
import argparse
import numpy as np
import mmcv
import copy

# CLASS_NAME = [
#     'toilet', 'bed', 'chair', 'sofa', 'dresser', 'table', 'cabinet',
#     'bookshelf', 'pillow', 'sink', 'bathtub', 'refrigerator', 'desk', 
#     'nightstand', 'counter', 'door', 'curtain', 'box', 'lamp', 'bag'
# ]
CAT_IDS = [33, 4, 5, 6, 17, 7, 3, 10, 18, 34, 36, 24, 14, 32, 12, 8, 16, 29, 35, 37]

# CAT_IDS2CLS = {
#     nyu40id: i
#     for i, nyu40id in enumerate(list())
# }

O365CLS = {"human": 0,
           "sneakers": 1,
           "chair": 2,
           "hat": 3,
           "lamp": 4,
           "bottle": 5,
           "cabinet/shelf": 6,
           "cup": 7,
           "car": 8,
           "glasses": 9,
           "picture/frame": 10,
           "desk": 11,
           "handbag": 12,
           "street lights": 13,
           "book": 14,
           "plate": 15,
           "helmet": 16,
           "leather shoes": 17,
           "pillow": 18,
           "glove": 19,
           "potted plant": 20,
           "bracelet": 21,
           "flower": 22,
           "monitor": 23,
           "storage box": 24,
           "plants pot/vase": 25,
           "bench": 26,
           "wine glass": 27,
           "boots": 28,
           "dining table": 29,
           "umbrella": 30,
           "boat": 31,
           "flag": 32,
           "speaker": 33,
           "trash bin/can": 34,
           "stool": 35,
           "backpack": 36,
           "sofa": 37,
           "belt": 38,
           "carpet": 39,
           "basket": 40,
           "towel/napkin": 41,
           "slippers": 42,
           "bowl": 43,
           "barrel/bucket": 44,
           "coffee table": 45,
           "suv": 46,
           "toy": 47,
           "tie": 48,
           "bed": 49,
           "traffic light": 50,
           "pen/pencil": 51,
           "microphone": 52,
           "sandals": 53,
           "canned": 54,
           "necklace": 55,
           "mirror": 56,
           "faucet": 57,
           "bicycle": 58,
           "bread": 59,
           "high heels": 60,
           "ring": 61,
           "van": 62,
           "watch": 63,
           "combine with bowl": 64,
           "sink": 65,
           "horse": 66,
           "fish": 67,
           "apple": 68,
           "traffic sign": 69,
           "camera": 70,
           "candle": 71,
           "stuffed animal": 72,
           "cake": 73,
           "motorbike/motorcycle": 74,
           "wild bird": 75,
           "laptop": 76,
           "knife": 77,
           "cellphone": 78,
           "paddle": 79,
           "truck": 80,
           "cow": 81,
           "power outlet": 82,
           "clock": 83,
           "drum": 84,
           "fork": 85,
           "bus": 86,
           "hanger": 87,
           "nightstand": 88,
           "pot/pan": 89,
           "sheep": 90,
           "guitar": 91,
           "traffic cone": 92,
           "tea pot": 93,
           "keyboard": 94,
           "tripod": 95,
           "hockey stick": 96,
           "fan": 97,
           "dog": 98,
           "spoon": 99,
           "blackboard/whiteboard": 100,
           "balloon": 101,
           "air conditioner": 102,
           "cymbal": 103,
           "mouse": 104,
           "telephone": 105,
           "pickup truck": 106,
           "orange": 107,
           "banana": 108,
           "airplane": 109,
           "luggage": 110,
           "skis": 111,
           "soccer": 112,
           "trolley": 113,
           "oven": 114,
           "remote": 115,
           "combine with glove": 116,
           "paper towel": 117,
           "refrigerator": 118,
           "train": 119,
           "tomato": 120,
           "machinery vehicle": 121,
           "tent": 122,
           "shampoo/shower gel": 123,
           "head phone": 124,
           "lantern": 125,
           "donut": 126,
           "cleaning products": 127,
           "sailboat": 128,
           "tangerine": 129,
           "pizza": 130,
           "kite": 131,
           "computer box": 132,
           "elephant": 133,
           "toiletries": 134,
           "gas stove": 135,
           "broccoli": 136,
           "toilet": 137,
           "stroller": 138,
           "shovel": 139,
           "baseball bat": 140,
           "microwave": 141,
           "skateboard": 142,
           "surfboard": 143,
           "surveillance camera": 144,
           "gun": 145,
           "Life saver": 146,
           "cat": 147,
           "lemon": 148,
           "liquid soap": 149,
           "zebra": 150,
           "duck": 151,
           "sports car": 152,
           "giraffe": 153,
           "pumpkin": 154,
           "Accordion/keyboard/piano": 155,
           "radiator": 156,
           "converter": 157,
           "tissue": 158,
           "carrot": 159,
           "washing machine": 160,
           "vent": 161,
           "cookies": 162,
           "cutting/chopping board": 163,
           "tennis racket": 164,
           "candy": 165,
           "skating and skiing shoes": 166,
           "scissors": 167,
           "folder": 168,
           "baseball": 169,
           "strawberry": 170,
           "bow tie": 171,
           "pigeon": 172,
           "pepper": 173,
           "coffee machine": 174,
           "bathtub": 175,
           "snowboard": 176,
           "suitcase": 177,
           "grapes": 178,
           "ladder": 179,
           "pear": 180,
           "american football": 181,
           "basketball": 182,
           "potato": 183,
           "paint brush": 184,
           "printer": 185,
           "billiards": 186,
           "fire hydrant": 187,
           "goose": 188,
           "projector": 189,
           "sausage": 190,
           "fire extinguisher": 191,
           "extension cord": 192,
           "facial mask": 193,
           "tennis ball": 194,
           "chopsticks": 195,
           "Electronic stove and gas st": 196,
           "pie": 197,
           "frisbee": 198,
           "kettle": 199,
           "hamburger": 200,
           "golf club": 201,
           "cucumber": 202,
           "clutch": 203,
           "blender": 204,
           "tong": 205,
           "slide": 206,
           "hot dog": 207,
           "toothbrush": 208,
           "facial cleanser": 209,
           "mango": 210,
           "deer": 211,
           "egg": 212,
           "violin": 213,
           "marker": 214,
           "ship": 215,
           "chicken": 216,
           "onion": 217,
           "ice cream": 218,
           "tape": 219,
           "wheelchair": 220,
           "plum": 221,
           "bar soap": 222,
           "scale": 223,
           "watermelon": 224,
           "cabbage": 225,
           "router/modem": 226,
           "golf ball": 227,
           "pine apple": 228,
           "crane": 229,
           "fire truck": 230,
           "peach": 231,
           "cello": 232,
           "notepaper": 233,
           "tricycle": 234,
           "toaster": 235,
           "helicopter": 236,
           "green beans": 237,
           "brush": 238,
           "carriage": 239,
           "cigar": 240,
           "earphone": 241,
           "penguin": 242,
           "hurdle": 243,
           "swing": 244,
           "radio": 245,
           "CD": 246,
           "parking meter": 247,
           "swan": 248,
           "garlic": 249,
           "french fries": 250,
           "horn": 251,
           "avocado": 252,
           "saxophone": 253,
           "trumpet": 254,
           "sandwich": 255,
           "cue": 256,
           "kiwi fruit": 257,
           "bear": 258,
           "fishing rod": 259,
           "cherry": 260,
           "tablet": 261,
           "green vegetables": 262,
           "nuts": 263,
           "corn": 264,
           "key": 265,
           "screwdriver": 266,
           "globe": 267,
           "broom": 268,
           "pliers": 269,
           "hammer": 270,
           "volleyball": 271,
           "eggplant": 272,
           "trophy": 273,
           "board eraser": 274,
           "dates": 275,
           "rice": 276,
           "tape measure/ruler": 277,
           "dumbbell": 278,
           "hamimelon": 279,
           "stapler": 280,
           "camel": 281,
           "lettuce": 282,
           "goldfish": 283,
           "meat balls": 284,
           "medal": 285,
           "toothpaste": 286,
           "antelope": 287,
           "shrimp": 288,
           "rickshaw": 289,
           "trombone": 290,
           "pomegranate": 291,
           "coconut": 292,
           "jellyfish": 293,
           "mushroom": 294,
           "calculator": 295,
           "treadmill": 296,
           "butterfly": 297,
           "egg tart": 298,
           "cheese": 299,
           "pomelo": 300,
           "pig": 301,
           "race car": 302,
           "rice cooker": 303,
           "tuba": 304,
           "crosswalk sign": 305,
           "papaya": 306,
           "hair dryer": 307,
           "green onion": 308,
           "chips": 309,
           "dolphin": 310,
           "sushi": 311,
           "urinal": 312,
           "donkey": 313,
           "electric drill": 314,
           "spring rolls": 315,
           "tortoise/turtle": 316,
           "parrot": 317,
           "flute": 318,
           "measuring cup": 319,
           "shark": 320,
           "steak": 321,
           "poker card": 322,
           "binoculars": 323,
           "llama": 324,
           "radish": 325,
           "noodles": 326,
           "mop": 327,
           "yak": 328,
           "crab": 329,
           "microscope": 330,
           "barbell": 331,
           "Bread/bun": 332,
           "baozi": 333,
           "lion": 334,
           "red cabbage": 335,
           "polar bear": 336,
           "lighter": 337,
           "mangosteen": 338,
           "seal": 339,
           "comb": 340,
           "eraser": 341,
           "pitaya": 342,
           "scallop": 343,
           "pencil case": 344,
           "saw": 345,
           "table tennis  paddle": 346,
           "okra": 347,
           "starfish": 348,
           "monkey": 349,
           "eagle": 350,
           "durian": 351,
           "rabbit": 352,
           "game board": 353,
           "french horn": 354,
           "ambulance": 355,
           "asparagus": 356,
           "hoverboard": 357,
           "pasta": 358,
           "target": 359,
           "hotair balloon": 360,
           "chainsaw": 361,
           "lobster": 362,
           "iron": 363,
           "flashlight": 364,}
TYPE2O365CLS = {v: k for k, v in O365CLS.items()}

CLASS_NAME = {"toilet": 0,
              "bed": 1,
              "chair": 2,
              "sofa": 3,
              "dresser": 4,
              "table": 5,
              "cabinet": 6,
              "bookshelf": 7,
              "pillow": 8,
              "sink": 9,
              "bathtub": 10,
              "refridgerator": 11,
              "desk": 12,
              "night stand": 13,
              "counter": 14,
              "door": 15,
              "curtain": 16,
              "box": 17,
              "lamp": 18,
              "bag": 19,}

O365CLS2TYPE = {O365CLS[t]: t for t in O365CLS}


def load_scannet_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_scene_ids(file_path):
    scenes = set()
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith('_pc.npy'):
                scene_id = '_'.join(file.split('_')[:3])
                scenes.add(scene_id)
    return list(scenes)

def init_info():
    info = {
        'point_cloud':{'num_features': 6, 'lidar_idx': ''},
        'pts_path': '',
        'pts_instance_mask_path': '',
        'pts_semantic_mask_path': '',
        'annos': {
            'gt_num': 0,
            'name': np.array([]),
            'location': np.array([]),
            'dimensions': np.array([]),
            'gt_boxes_upright_depth': np.array([]),
            'unaligned_location': np.array([]),
            'unaligned_dimensions': np.array([]),
            'unaligned_gt_boxes_upright_depth': np.array([]),
            'index': np.array([]),
            'class': np.array([]),
            'axis_align_matrix': np.eye(4)
        }
    }
    return info

def save_scannet_pkl(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def processing_info(scene_ids, pl_path):
    infos = []
    for scene_id in scene_ids:
        info = init_info()
        print(f'Processing {scene_id}...\n')
        info['point_cloud']['lidar_idx'] = scene_id
        info['pts_path'] = osp.join('points', f'{scene_id}.bin')
        tmp_bboxes = np.load(osp.join(pl_path, 'labels', f'{scene_id}_bbox.npy'))
        tmp_pose = np.loadtxt(osp.join(pl_path, 'calibs', f'{scene_id}_pose.txt'))
        info['annos']['gt_num'] = tmp_bboxes.shape[0]
        if info['annos']['gt_num'] != 0:
            labels = tmp_bboxes[:, -1]
            names = [TYPE2O365CLS[label] for label in labels]
            classes = [O365CLS[name] for name in names]
            info['annos']['name'] = np.array(names)
            info['annos']['location'] = tmp_bboxes[:, :3]
            info['annos']['dimensions'] = tmp_bboxes[:, 3:6]
            info['annos']['gt_boxes_upright_depth'] = tmp_bboxes[:, :6]
            info['annos']['unaligned_location'] = tmp_bboxes[:, :3]
            info['annos']['unaligned_dimensions'] = tmp_bboxes[:, 3:6]
            info['annos']['unaligned_gt_boxes_upright_depth'] = tmp_bboxes[:, :6]
            info['annos']['index'] = np.arange(len(names))
            info['annos']['class'] = np.array(classes)
        info['annos']['axis_align_matrix'] = tmp_pose
        infos.append(copy.deepcopy(info))
    return infos

def main():
    '''{'point_cloud': {
        'num_features': 6, 
        'lidar_idx': 'scene0191_00'}, 

    'pts_path': 'points/scene0191_00.bin', 
    'pts_instance_mask_path': 'instance_mask/scene0191_00.bin', 
    'pts_semantic_mask_path': 'semantic_mask/scene0191_00.bin', 
    'annos': {
        'gt_num': 3, 
        'name': array(['door', 'table', 'chair'], dtype='<U5'), 
        'location': array([[ 1.9491133 ,  2.5492249 ,  1.07194896],
                        [ 1.06101452, -0.36034763,  0.43245818],
                        [ 1.27606464, -0.55551459,  0.26375593]]), 
        'dimensions': array([[0.17460823, 1.10707517, 1.39844283],
                        [1.69858453, 0.86210089, 0.87308059],
                        [0.64525549, 0.43832136, 0.47023378]]), 
        'gt_boxes_upright_depth': array([[ 1.9491133 ,  2.5492249 ,  1.07194896,  0.17460823,  1.10707517,1.39844283],
        [ 1.06101452, -0.36034763,  0.43245818,  1.69858453,  0.86210089,0.87308059],
        [ 1.27606464, -0.55551459,  0.26375593,  0.64525549,  0.43832136,0.47023378]]), 
        'unaligned_location': array([[0.69672561, 3.18364525, 1.14261401],[3.69625235, 2.92570305, 0.50312316], [3.86308336, 3.1846931 , 0.33442092]]), 
        'unaligned_dimensions': array([[1.08603287, 0.28095126, 1.39844286], [1.09702015, 1.7518568 , 0.87308061], [0.48175812, 0.6021471 , 0.4702338 ]]), 
        'unaligned_gt_boxes_upright_depth': array([[0.69672561, 3.18364525, 1.14261401, 1.08603287, 0.28095126, 1.39844286], [3.69625235, 2.92570305, 0.50312316, 1.09702015, 1.7518568 ,
        0.87308061], [3.86308336, 3.1846931 , 0.33442092, 0.48175812, 0.6021471 , 0.4702338 ]]), 
        'index': array([0, 1, 2], dtype=int32), 
        'class': array([15,  5,  2]), 
        'axis_align_matrix': array([[-0.21644 ,  0.976296,  0.      , -1.01457 ],
        [-0.976296, -0.21644 ,  0.      ,  3.91808 ],
        [ 0.      ,  0.      ,  1.      , -0.070665],
        [ 0.      ,  0.      ,  0.      ,  1.      ]])}}
    '''
    # annos_train_path = f'{file_path}/scannet_infos_train.pkl'
    # annos_val_path = f'{file_path}/scannet_infos_val.pkl'
    pl_train_path = './data/ov-det/ScanNet_processed'
    pl_val_path = './data/ov-det/ScanNet_processed'
    save_train_path = './data/ov-det/ScanNet_processed/scannet_processed_infos_train.pkl'
    save_val_path = './data/ov-det/ScanNet_processed/scannet_processed_infos_val.pkl'
    scannet_train_scene_ids = np.loadtxt(osp.join(pl_train_path, 'ImageSets/train.txt'), dtype=str)
    scannet_val_scene_ids = np.loadtxt(osp.join(pl_val_path, 'ImageSets/val.txt'), dtype=str)

    print('Processing train ...')
    train_infos = processing_info(scannet_train_scene_ids, pl_train_path)
    print('Processing val ...')
    val_infos = processing_info(scannet_val_scene_ids, pl_val_path)

    save_scannet_pkl(train_infos, save_train_path)
    save_scannet_pkl(val_infos, save_val_path)
    print('Finished!')


if __name__ == '__main__':
    main()