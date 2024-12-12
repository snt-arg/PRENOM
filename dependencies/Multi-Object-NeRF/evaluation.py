import os
import json
import trimesh
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree as KDTree

def completion_ratio(gt_points, rec_points, dist_th=0.01):
    gen_points_kd_tree = KDTree(rec_points)
    one_distances, _ = gen_points_kd_tree.query(gt_points)
    completion = np.mean((one_distances < dist_th).astype(np.float))
    return completion


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, _ = gt_points_kd_tree.query(rec_points)
    gen_to_gt_chamfer = np.mean(two_distances)
    return gen_to_gt_chamfer


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    one_distances, _ = gt_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(one_distances)
    return gt_to_gen_chamfer


def chamfer(gt_points, rec_points):
    # one direction
    gen_points_kd_tree = KDTree(rec_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(one_distances)

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(rec_points)
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

    # accuracy_rec *= 100  # convert to cm
    # completion_rec *= 100  # convert to cm
    # completion_ratio_rec *= 100  # convert to %
    # print('accuracy: ', accuracy_rec)
    # print('completion: ', completion_rec)
    # print('completion ratio: ', completion_ratio_rec)
    # print("completion_ratio_rec_1cm ", completion_ratio_rec_1)
    metrics[0].append(accuracy_rec)
    metrics[1].append(completion_rec)
    metrics[2].append(completion_ratio_rec_1)
    metrics[3].append(completion_ratio_rec)
    return metrics

if __name__ == "__main__":
    ply = "output/0.ply"
    gt_ply = "/home/saadejazz/sap_nerf/output/room/gt_mesh/laptop.ply"
    
    ply = trimesh.load(ply)
    gt_ply = trimesh.load(gt_ply)
    
    # transform the gt_ply to the same coordinate system as sapiens
    T = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    gt_ply.apply_transform(T)
    gt_ply.vertices[:, 0] -= 0.085571
    gt_ply.vertices[:, 1] -= 0.0106116
    gt_ply.vertices[:, 2] += 0.490172
    
    # # original paper has a different coordinate system
    # gt_ply.vertices[:, 0] -= 0.0106116
    # gt_ply.vertices[:, 1] += 0.085571
    # gt_ply.vertices[:, 2] += 0.420172
    
    # save the new gt_ply
    gt_ply.export("output/1_gt.ply")
    
    metrics = calc_3d_metric(ply, gt_ply)
    
    print("Accuracy: ", metrics[0])
    print("Completion: ", metrics[1])
    print("Completion Ratio 1cm: ", metrics[2])
    print("Completion Ratio 0.4cm: ", metrics[3])