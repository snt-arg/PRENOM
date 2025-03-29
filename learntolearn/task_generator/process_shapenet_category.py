import multiprocessing as mp
from tqdm import tqdm
import random
import trimesh
import json
import os


### change these configuration based on the ShapeNet category ###

# the location with the ShapeNetCore.v2_normalized dataset
category = "chair"
directory = f"/home/saadejazz/shapenet/data/ShapeNetCore.v2_normalized/03001627/"

# the average aspect ratios of the objects in the category
# objects with aspect ratios differing more than max_diff from the average will be rejected
# this is just to ensure that the objects are more or less similar (some objects in ShapeNet are very weird)
# tune this filter to get the desired number of objects
average_dim = [0.5, 0.5, 1.1]
avg_xy = average_dim[0]/average_dim[1]
avg_yz = average_dim[1]/average_dim[2]
avg_zx = average_dim[2]/average_dim[0]
max_diff = 1/3.0

# minimum number of vertices in the ground truth mesh
min_verts = 5000

# the transformation from the frame in ShapeNet to your desired frame
T = [[0, 0, 1, 0],
     [1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]]

# the proportion of models for testing
TRAIN_TEST_SPLIT = 0.95

# multiprocessing
NUM_PROCESSES = 8

### end of configuration ###


def process_files(files):
    # [TODO] - could look into tqdm-multiprocess (not critical)
    for file in tqdm(files):
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
        if num_verts < min_verts:
            continue
        
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
        if abs(xy - avg_xy) + abs(yz - avg_yz) + abs(zx - avg_zx) > max_diff * 3:
            continue

        # save the mesh and bounding box
        rand = random.random()
        train_test = "train" if rand < TRAIN_TEST_SPLIT else "test"
        save_dir = os.path.join("cads", category, train_test, file)
        os.makedirs(save_dir, exist_ok=True)
        mesh.export(f'{save_dir}/object.obj')
        with open(os.path.join(save_dir, "bounding_box.json"), "w") as f:
            json.dump(bbox, f) 


if __name__ == "__main__":
    # get the list of files in the directory
    files = os.listdir(directory)
    files = [file for file in files if os.path.isdir(os.path.join(directory, file))]

    # process the files in parallel
    with mp.Pool(NUM_PROCESSES) as pool:
        pool.map(process_files, [files[i::NUM_PROCESSES] for i in range(NUM_PROCESSES)])

    print("Processing finished")