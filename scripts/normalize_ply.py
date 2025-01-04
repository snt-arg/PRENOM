import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    category = "mug"
    pointcloud_file = f"/home/saadejazz/RO-MAP-NG/dependencies/Multi-Object-NeRF/output/meta_model.ply"
    
    # The pose of the object and the extent of the object
    with open(f"/home/saadejazz/sap_nerf/output/{category}/obj_offline/0.txt" , "r") as f:
        # ignore the first line (comment) and the first element of the second line (category id)
        lines = f.readlines()[1:]
        category_id = int(lines[0].split()[0])
        tx, ty, tz, qx, qy, qz, qw, a1, a2, a3, a4, a5, a6 = [float(x) for x in lines[0].split()[1:]]
        print(tx, ty, tz, qx, qy, qz, qw)
    
    # # Two - pose as 4x4 matrix
    # Two = np.eye(4)
    # Two[:3, 3] = [tx, ty, tz]
    # Two[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    
    # The extent of the object
    bbox_min = np.array([a1, a2, a3])
    bbox_max = np.array([a4, a5, a6])
    print(bbox_min, bbox_max)
    
    # Normalize the pointcloud
    pc = trimesh.load(pointcloud_file)
    pc.vertices = (pc.vertices - bbox_min) / (bbox_max - bbox_min)
    print(pc.vertices.min(axis=0), pc.vertices.max(axis=0))

    # # apply rotation because trimesh messes up the normals
    # T_tri = np.array([
    #     [1, 0, 0, 0],
    #     [0, 0, -1, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1]
    # ])
    # pc.apply_transform(T_tri)
    
    # Save the normalized pointcloud
    pc.export(f"{category_id}.ply")