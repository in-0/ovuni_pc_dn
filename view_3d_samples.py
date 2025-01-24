import open3d as o3d
import numpy as np
import argparse
import os
import os.path as osp
import cv2

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


def visualize_npy(file_path, pose):
    # Load .npy file
    point_cloud = np.load(file_path)
    if pose is not None:
        pose = np.loadtxt(pose)
        # pose_inv = np.linalg.inv(pose)
        pts = np.ones((point_cloud.shape[0], 4))
        pts[:, 0:3] = point_cloud[:, 0:3]
        pts = np.dot(pts, pose.transpose())
        point_cloud = np.concatenate([pts[:, 0:3], point_cloud[:, 3:]], axis=1)
    
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    
    if point_cloud.shape[1] == 6:  # If the file contains color information
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6] / 255.0)
    
    # Visualize
    # o3d.visualization.draw_geometries([pcd])
    return pcd

def visualize_bin(file_path, pose):
    # Load .bin file
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    if pose is not None:
        pose = np.loadtxt(pose)
        pose_inv = np.linalg.inv(pose)
        pts = np.ones((point_cloud.shape[0], 4))
        pts[:, 0:3] = point_cloud[:, 0:3]
        pts = np.dot(pts, pose_inv.transpose())  # Nx4
        point_cloud = np.concatenate([pts[:, 0:3], point_cloud[:, 3:]], axis=1)
    # point_cloud[:, 0] = -point_cloud[:, 0]  # Flip x-axis
    
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

    if point_cloud.shape[1] == 6:  # If the file contains color information
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6] / 255.0)
    
    # Visualize
    # o3d.visualization.draw_geometries([pcd])
    return pcd

def visualize_bboxes(bboxes):
    # bboxes = np.load(bbox_file_path)
    geometries = []
    
    for bbox in bboxes:
        center = bbox[:3]
        extents = bbox[3:6]
        if bbox.shape[-1] == 7:
            yaw = bbox[6]
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
        elif bbox.shape[-1] == 8:
            yaw = bbox[6]
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
            label = bbox[7]
            class_name = TYPE2O365CLS.get(label, "Unknown")
        
        obb = o3d.geometry.OrientedBoundingBox(center, np.eye(3), extents)
        obb.color = (1, 0, 0)
        geometries.append(obb)

    return geometries

def is_axis_aligned_pca(points):
    # Compute PCA
    pca = PCA(n_components=3)
    pca.fit(points[:, :3])
    
    # Get principal components
    principal_axes = pca.components_
    
    # Check alignment with axis directions
    axis_directions = np.eye(3)
    for axis in axis_directions:
        if np.any(np.allclose(principal_axes, axis)):
            return True
    return False


def arg_parse():
    parser = argparse.ArgumentParser(description="Visualize Scannet files")
    parser.add_argument("--pt-path", type=str, help="Path to .pkl file")
    parser.add_argument("--pose-path", type=str, default=None)
    parser.add_argument("--bbox-path", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    pt_path = args.pt_path
    bbox_path = args.bbox_path
    pose_path = args.pose_path

    # point_cloud = np.load(pt_path)
    # bboxes = np.load(bbox_path)
    # axis_align_matrix = np.loadtxt(axis_align_matrix_path)
    # is_axis_aligned_pca(point_cloud)
    # breakpoint()



    # Load pt and bbox files
    if pt_path.endswith(".npy"):
        pcd = visualize_npy(pt_path, pose_path)
    elif pt_path.endswith(".bin"):
        pcd = visualize_bin(pt_path, pose_path)

    # Create a coordinate frame at the origin
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    if bbox_path:
        bboxes = np.load(bbox_path)
        bboxes = visualize_bboxes(bboxes)
    
    # Visualize point cloud with origin
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(origin)
    if bbox_path:
        for bbox in bboxes:
            vis.add_geometry(bbox)
    vis.get_render_option().line_width = 30
    vis.run()