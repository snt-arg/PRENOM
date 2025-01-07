import trimesh
import numpy as np
from scipy.spatial import cKDTree as KDTree
from pyntcloud import PyntCloud

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

if __name__ == "__main__":
    ply = "/home/saadejazz/RO-MAP-NG/dependencies/Multi-Object-NeRF/output/999_0.ply"
    gt_ply = "/home/saadejazz/RO-MAP-NG/room/gt_mesh/laptop.ply"
    
    gt_ply = PyntCloud.from_file(gt_ply)
    gt_ply = gt_ply.get_sample("mesh_random", n=64*64*64, rgb=False, normals=False)
    gt_ply = trimesh.Trimesh(vertices=gt_ply[["x", "y", "z"]].values)
    
    ply = trimesh.load(ply)
    
    print(len(gt_ply.vertices))
    
    # transform the gt_ply to the same coordinate system as sapiens
    T = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    gt_ply.apply_transform(T)
    gt_ply.vertices[:, 0] += 0.085571
    gt_ply.vertices[:, 1] -= 0.0106116
    gt_ply.vertices[:, 2] += 0.405757
    
    # # original paper has a different coordinate system
    # gt_ply.vertices[:, 0] -= 0.0106116
    # gt_ply.vertices[:, 1] += 0.085571
    # gt_ply.vertices[:, 2] += 0.420172
    
    # save the new gt_ply
    gt_ply.export("output/1_gt.ply")
    ply.export("output/10.ply")
    
    metrics = calc_3d_metric(ply, gt_ply)
    
    print("Accuracy: ", metrics[0])
    print("Completion: ", metrics[1])
    print("Completion Ratio 1cm: ", metrics[2])
    print("Completion Ratio 0.4cm: ", metrics[3])