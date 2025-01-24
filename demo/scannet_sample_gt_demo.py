# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet3d.apis import inference_detector, init_model, show_result_meshlab
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose

from copy import deepcopy
import numpy as np

from mmcv.parallel import collate, scatter

def loading_data(model, pcd):
    cfg = model.cfg

    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)

    # load from point clouds file
    data = dict(
        pts_filename=pcd,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        # for ScanNet demo we need axis_align_matrix
        ann_info=dict(axis_align_matrix=np.eye(4)),
        sweeps=[],
        # set timestamp = 0
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    # this is a workaround to avoid the bug of MMDataParallel
    data['img_metas'] = data['img_metas'].data
    data['points'] = data['points'].data
    breakpoint()


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()


    # # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # Load the point cloud and bboxes from the pkl file
    bboxes, data = loading_data(model, args.pcd)
    # show the results
    show_result_meshlab(
        data,
        bboxes,
        args.out_dir,
        args.score_thr,
        show=args.show,
        snapshot=args.snapshot,
        task='det')


if __name__ == '__main__':
    main()
