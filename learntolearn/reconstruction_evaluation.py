import trimesh
import numpy as np
from scipy.spatial import cKDTree as KDTree


def completion_ratio(gt_points, rec_points, dist_th=0.01):
    gen_points_kd_tree = KDTree(rec_points)
    one_distances, _ = gen_points_kd_tree.query(gt_points, workers=-1)
    completion = np.mean((one_distances < dist_th).astype(np.float))
    return completion

def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, _ = gt_points_kd_tree.query(rec_points, workers=-1)
    gen_to_gt_chamfer = np.mean(two_distances)
    return gen_to_gt_chamfer


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    one_distances, _ = gt_points_kd_tree.query(gt_points, workers=-1)
    gt_to_gen_chamfer = np.mean(one_distances)
    return gt_to_gen_chamfer


def chamfer(gt_points, rec_points):
    # one direction
    gen_points_kd_tree = KDTree(rec_points)
    one_distances, _ = gen_points_kd_tree.query(gt_points, workers=-1)
    gt_to_gen_chamfer = np.mean(one_distances)

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, _ = gt_points_kd_tree.query(rec_points, workers=-1)
    gen_to_gt_chamfer = np.mean(two_distances)

    return (gt_to_gen_chamfer + gen_to_gt_chamfer) / 2.

def calc_3d_metric(ply, ply_gt):
    """
    3D reconstruction metric.
    """
    metrics = [[] for _ in range(4)]
    
    rec_pc_tri = trimesh.PointCloud(vertices=ply.vertices)
    gt_pc_tri = trimesh.PointCloud(vertices=ply_gt.vertices)
    
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, 0.004)
    completion_ratio_rec_1 = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, 0.01)

    metrics[0].append(accuracy_rec)
    metrics[1].append(completion_rec)
    metrics[2].append(completion_ratio_rec_1)
    metrics[3].append(completion_ratio_rec)
    return metrics
