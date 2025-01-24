# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import warnings
from os import path as osp

import numpy as np

from mmdet3d.core import instance_seg_eval, show_result, show_seg_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmseg.datasets import DATASETS as SEG_DATASETS
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets import ScanNetDataset
from mmdet3d.datasets.pipelines import Compose


@DATASETS.register_module()
class ScanNetDataset_OV_processed(ScanNetDataset):
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
    TYPE2CLS = {v: k for k, v in CLASS_NAME.items()}

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 novel_classes=None,
                 modality=dict(use_camera=False, use_depth=True),
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 **kwargs):
        self.CLASS_NAME = novel_classes
        self.classes = tuple(self.O365CLS.keys())
        self.novel_classes = tuple(self.CLASS_NAME.keys())
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=self.classes,
            # novel_classes=self.novel_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert 'use_camera' in self.modality and \
               'use_depth' in self.modality
        assert self.modality['use_camera'] or self.modality['use_depth']

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - img_prefix (str, optional): Prefix of image files.
                - img_info (dict, optional): Image info.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])
        input_dict = dict(sample_idx=sample_idx)

        if self.modality['use_depth']:
            input_dict['pts_filename'] = pts_filename
            input_dict['file_name'] = pts_filename

        if self.modality['use_camera']:
            img_info = []
            for img_path in info['img_paths']:
                img_info.append(
                    dict(filename=osp.join(self.data_root, img_path)))
            intrinsic = info['intrinsics']
            axis_align_matrix = self._get_axis_align_matrix(info)
            depth2img = []
            for extrinsic in info['extrinsics']:
                depth2img.append(
                    intrinsic @ np.linalg.inv(axis_align_matrix @ extrinsic))

            input_dict['img_prefix'] = None
            input_dict['img_info'] = img_info
            input_dict['depth2img'] = depth2img

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
                - axis_align_matrix (np.ndarray): Transformation matrix for
                    global scene alignment.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)

        # pts_instance_mask_path = osp.join(self.data_root,
        #                                   info['pts_instance_mask_path'])
        # pts_semantic_mask_path = osp.join(self.data_root,
        #                                   info['pts_semantic_mask_path'])

        axis_align_matrix = self._get_axis_align_matrix(info)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            # pts_instance_mask_path=pts_instance_mask_path,
            # pts_semantic_mask_path=pts_semantic_mask_path,
            axis_align_matrix=axis_align_matrix)
        return anns_results

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(type='GlobalAlignment', rotation_axis=2),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            pred_bboxes = result['boxes_3d'].tensor.numpy()
            show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name,
                        show)

    def show_gt(self, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        pipeline = self._get_pipeline(pipeline)
        data_info = self.data_infos[0]
        pts_path = data_info['pts_path']
        file_name = osp.split(pts_path)[-1].split('.')[0]
        points = self._extract_data(0, pipeline, 'points').numpy()
        gt_bboxes = self.get_ann_info(0)['gt_bboxes_3d'].tensor.numpy()
        pred_bboxes = []
        show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name, show)

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Defaults to None.
            iou_thr (list[float]): AP IoU thresholds. Defaults to (0.25, 0.5).
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        from ..core.indoor_eval import indoor_eval_ov
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = [info['annos'] for info in self.data_infos]
        label2cat = self.TYPE2CLS
        ret_dict = indoor_eval_ov(
            gt_annos,
            results,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d)
        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return ret_dict
