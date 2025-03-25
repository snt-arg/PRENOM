import trimesh
import json
import os


### change these configuration based on the ShapeNet category ###

# the location with the ShapeNetCore.v2_normalized dataset
category = "chair"
directory = f"/home/saadejazz/meta_nerf/shapenet/data/ShapeNetCore.v2_normalized/03001627/"

# the average aspect ratios of the objects in the category
# objects with aspect ratios differing more than max_diff from the average will be rejected
# this is just to ensure that the objects are more or less similar (some objects in ShapeNet are very weird)
# tune this filter to get the desired number of objects
average_dim = [0.5, 0.5, 1.1]
avg_xy = average_dim[0]/average_dim[1]
avg_yz = average_dim[1]/average_dim[2]
avg_zx = average_dim[2]/average_dim[0]
max_diff = 0.333

# minimum number of vertices in the ground truth mesh
min_verts = 5000

# the transformation from the frame in ShapeNet to your desired frame
T = [[0, 0, 1, 0],
     [1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]]

### end of configuration ###


chosen = 0
rejected = 0
files = os.listdir(directory)
for file in files:
    # load the mesh
    dir = os.path.join(directory, file, "models", "model_normalized.obj")
    mesh = trimesh.load(dir, force="mesh")

    # get the bbox of the mesh
    bbox = mesh.bounds.copy()
    center = (bbox[0] + bbox[1]) / 2
    
    # offset the center to the origin of the mesh
    for vert in mesh.vertices:
        vert -= center
    
    bbox = mesh.bounds.copy()
    center = (bbox[0] + bbox[1]) / 2
    num_verts = len(mesh.vertices)
    print(f"Number of vertices: {num_verts}")
    
    # transformation
    mesh.apply_transform(T)
    bbox = mesh.bounds.copy()
    bbox = {
        "min": bbox[0].tolist(),
        "max": bbox[1].tolist()
    }

    # check if the aspect ratios don't differ too much from average
    xy = bbox["max"][0]/bbox["max"][1]
    yz = bbox["max"][1]/bbox["max"][2]
    zx = bbox["max"][2]/bbox["max"][0]
    print(f"Aspect ratios: {xy}, {yz}, {zx}")
    print("Average aspect ratios: ", avg_xy, avg_yz, avg_zx)
    if abs(xy - avg_xy) + abs(yz - avg_yz) + abs(zx - avg_zx) > max_diff * 3 or num_verts < min_verts:
        print(f"Aspect ratio of {file} is not close to average")
        rejected += 1
        continue

    # save the mesh and bounding box
    chosen += 1
    save_dir = os.path.join(category, file)
    os.makedirs(save_dir, exist_ok=True)
    mesh.export(f'{save_dir}/object.obj')
    with open(os.path.join(save_dir, "bounding_box.json"), "w") as f:
        json.dump(bbox, f) 

print("Chosen: ", chosen)
print("Rejected: ", rejected)