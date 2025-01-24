import open3d as o3d
import numpy as np
import argparse
import os
import os.path as osp
import cv2
import pickle

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_pkl(file_path):
    """
    Load a .pkl file.

    Args:
        file_path (str): Path to the .pkl file.

    Returns:
        dict: The content of the .pkl file.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def visualize_bin(file_path, pose):
    # Load .bin file
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 3)
    if pose is not None:
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
    parser.add_argument("--pkl-path", type=str, help="Path to .pkl file")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    pkl_path = args.pkl_path
    sunrgbd_infos = load_pkl(pkl_path)
    scene_idx = 7303
    sample = None
    for info in sunrgbd_infos:
        if info['point_cloud']['lidar_idx'] == scene_idx:
            sample = info
            break
    
    root_dir = './data/sunrgbd'
    pt_path = sample['pts_path']
    bboxes = sample['annos']['gt_boxes_upright_depth']
    pcd = visualize_bin(osp.join(root_dir, pt_path), None)
    bboxes = visualize_bboxes(bboxes)
    breakpoint()

    # Load pt and bbox files
    # pt_path.endswith(".bin"):
    #     pcd = visualize_bin(pt_path)

    # Create a coordinate frame at the origin
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # if bbox_path:
    #     bboxes = np.load(bbox_path)
    #     bboxes = visualize_bboxes(bboxes)
    
    # Visualize point cloud with origin
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(origin)
    for bbox in bboxes:
        vis.add_geometry(bbox)
    vis.get_render_option().line_width = 30
    vis.run()