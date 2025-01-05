import subprocess
import json
import numpy as np
import os
import time
import shutil
import trimesh
from pyntcloud import PyntCloud
import random

from .reconstruction_evaluation import accuracy, completion

SAPIENS_DATA_DIR = "/home/saadejazz/sap_nerf/output"
TRAINING_DIR = "/home/saadejazz/RO-MAP-NG/dependencies/Multi-Object-NeRF"
# output directory for the trained models (needs to be the same in system.json of the training script)
# must have a trailing slash
OUTPUT_DIR = "/home/saadejazz/momvml_output/"

# other constants
LAMBDA_MODEL_SIZE = 0.1

def single_train_call(
    base_path,
    system_path,
    data_dir
):
    # call the object training script and wait for it to finish
    command = f"cd {TRAINING_DIR} && __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./build/OfflineNeRF {base_path} {system_path} {data_dir}"
    object_training = subprocess.Popen(command, shell=True)
    object_training.wait()


def run_single_meta_iteration(
    category,
    num_meta_loops,
    num_inner_iterations,
    meta_lr,
    inner_lr,
    log2_hashmap_size,
    per_level_scale,
    n_neurons,
    n_hidden_layers
):
    """
    Run a single meta iteration.
    
    Args:
    - category (str): The category of the object.
    - num_meta_loops (int): The number of meta loops: (20 - 100)
    - num_inner_iterations (int): The number of iterations of the inner loop: (100 - 1500)
    - meta_lr (float): The meta learning rate: (1e-3 - 1e-1)
    - inner_lr (float): The inner learning rate: (1e-3 - 1e-1)
    - log2_hashmap_size (int): The log2 of the hashmap size: (14 - 24)
    - per_level_scale (float): The per level scale: (1.26 - 2.00)
    - n_neurons (int): The number of neurons: (16, 32, 64, 128)
    - n_hidden_layers (int): The number of hidden layers: (1 - 3)    
    """
    
    # create an identifier for this run
    identifier = np.random.randint(0, 10000000)
    
    # get all the available data within the category
    # [TODO] - use an Object-Oriented approach to get the entire dataset once
    data = os.listdir(os.path.join(SAPIENS_DATA_DIR, category, "train"))
    
    # create a base json file
    with open(os.path.join(TRAINING_DIR, 'Core/configs/base.json'), 'r') as file:
        base = json.load(file)
        
    new_base = base.copy()
    new_base["optimizer"]["nested"]["nested"]["learning_rate"] = inner_lr
    new_base["meta_optimizer"]["nested"]["nested"]["learning_rate"] = meta_lr
    new_base["encoding"]["log2_hashmap_size"] = log2_hashmap_size
    new_base["encoding"]["per_level_scale"] = per_level_scale
    new_base["network"]["n_neurons"] = n_neurons
    new_base["network"]["n_hidden_layers"] = n_hidden_layers
    
    # save the new json file
    base_path = os.path.join(OUTPUT_DIR, f'base_{identifier}.json')
    with open(base_path, 'w') as file:
        json.dump(new_base, file)
        
    # create a system json file
    new_system = {
        "use_depth": True,
        "do_meta": True,
        
        "object_indices": [0],
        
        "save_model": True,
        "save_identifier": identifier,
        
        "output_dir": OUTPUT_DIR,
        
        "load_model": False,
        "load_path": "",
        
        "visualize": False,
        
        "n_iters_per_step": num_inner_iterations,
        "n_steps": 1,
        
        "meta_n_iters_per_step": num_inner_iterations,
        "meta_n_steps": 1,
        "meta_n_loops": 1
    }
    
    system_path = os.path.join(OUTPUT_DIR, f'system_{identifier}.json')
    with open(system_path, 'w') as file:
        json.dump(new_system, file)
    
    # call the single call function to get a starting model
    data_dir = random.choice(data)
    data_dir = os.path.join(SAPIENS_DATA_DIR, category, "train", data_dir)
    single_train_call(base_path, system_path, data_dir)
    
    # now change the new_system to load the generated model
    new_system["load_model"] = True
    new_system["load_path"] = os.path.join(OUTPUT_DIR, f"{identifier}.json")
    with open(system_path, 'w') as file:
        json.dump(new_system, file)
        
    # run the meta loop
    for _ in range(num_meta_loops - 1):
        data_dir = random.choice(data)
        data_dir = os.path.join(SAPIENS_DATA_DIR, category, "train", data_dir)
        single_train_call(base_path, system_path, data_dir)
        
    # delete the json files
    os.remove(base_path)
    os.remove(system_path)
    return identifier
  
def evaluate_run(
    category,
    identifier,
    inner_lr,
    log2_hashmap_size,
    per_level_scale,
    n_neurons,
    n_hidden_layers
):
    # call the object evaluation script and wait for it to finish
    inner_loops_to_test = [
         50,
        100,
        200,
        # 300,
    ]
    use_depths = [True, False]
    
    # get all the test sets
    test_dir = os.path.join(SAPIENS_DATA_DIR, category, "test")
    test_sets = os.listdir(test_dir)
    print(test_sets)
    
    # set a new base config
    with open(f'{TRAINING_DIR}/Core/configs/base.json', 'r') as file:
        base = json.load(file)
        
    new_base = base.copy()
    new_base["optimizer"]["nested"]["nested"]["learning_rate"] = inner_lr
    new_base["encoding"]["log2_hashmap_size"] = log2_hashmap_size
    new_base["encoding"]["per_level_scale"] = per_level_scale
    new_base["network"]["n_neurons"] = n_neurons
    new_base["network"]["n_hidden_layers"] = n_hidden_layers
    
    # save the new json file
    base_path = os.path.join(OUTPUT_DIR, f'base_{identifier}.json')
    with open(base_path, 'w') as file:
        json.dump(new_base, file)
    
    # set the system config
    model_path = os.path.join(OUTPUT_DIR, f"{identifier}.json")
    new_system = {
        "do_meta": False,
        
        "object_indices": [0],
        
        "save_model": True,
        "save_identifier": identifier,
        
        "output_dir": OUTPUT_DIR,
        
        "load_model": True,
        "load_path": model_path,
        
        "visualize": False,
        
        "n_steps": 1,
        
        "meta_n_iters_per_step": 1,
        "meta_n_steps": 1,
        "meta_n_loops": 1
    }
    system_path = os.path.join(OUTPUT_DIR, f'system_{identifier}.json')
    with open(system_path, 'w') as file:
        json.dump(new_system, file)
  
    accuracies = []
    completions = []
  
    tic = time.time()
  
    for depth_flag in use_depths:
        new_system["use_depth"] = depth_flag
        depth_accuracies = []
        depth_completions = []
        
        for inner_loops in inner_loops_to_test:
            new_system["n_iters_per_step"] = inner_loops
            with open(system_path, 'w') as file:
                json.dump(new_system, file)
            
            for test_set in test_sets:
                data_dir = os.path.join(test_dir, test_set)
                single_train_call(base_path, system_path, data_dir)
                
                # get the scale of the object
                with open(os.path.join(test_dir, test_set, "obj_offline/scale.txt"), 'r') as file:
                    line = file.readlines()[0]
                    scale = float(line.strip())
                
                # get the resulting mesh
                ret_pc_path = os.path.join(OUTPUT_DIR, f"{identifier}_0.ply")
                gt_pc = os.path.join(test_dir, test_set, "0.obj")
                gt_pc = trimesh.load(gt_pc, force='mesh')
                gt_pc.vertices = gt_pc.vertices * scale
                new_gt_pc_path = os.path.join(OUTPUT_DIR, f"{identifier}_gt.ply")
                gt_pc.export(new_gt_pc_path)
                gt_pc = PyntCloud.from_file(new_gt_pc_path)
                gt_pc = gt_pc.get_sample("mesh_random", n=64*64*64, rgb=False, normals=False)
                gt_pc = trimesh.Trimesh(vertices=gt_pc[["x", "y", "z"]].values)
                ret_pc = trimesh.load(ret_pc_path, force='mesh')
                accuracy_metric = accuracy(gt_pc.vertices, ret_pc.vertices)
                completion_metric = completion(gt_pc.vertices, ret_pc.vertices)
                depth_accuracies.append(accuracy_metric)
                depth_completions.append(completion_metric)
        
        print("Depth flag: ", depth_flag)
        print("Mean accuracy: ", np.mean(depth_accuracies))
        print("Mean completion: ", np.mean(depth_completions), "\n")
        
        accuracies.extend(depth_accuracies)
        completions.extend(depth_completions)
    
    toc = time.time()
    
    print("Final results:")
    print("Mean accuracy: ", np.mean(accuracies))
    print("Mean completion: ", np.mean(completions))
    first_objective = (np.mean(accuracies) + np.mean(completions))/2
  
    # remove the json files
    os.remove(base_path)
    os.remove(system_path)
  
    # second objective is the size of the saved model and training time
    model_size = os.path.getsize(model_path)
    model_size = model_size / (1024 * 1024) # in MB
    print("Model size: ", model_size)
    
    total_iterations = np.sum(inner_loops_to_test) * len(use_depths) * len(test_sets)
    time_per_iteration = ((toc - tic) * 1000) / total_iterations # in ms
    print("Time per iteration: ", time_per_iteration)
    
    second_objective = LAMBDA_MODEL_SIZE * model_size + time_per_iteration
    print("Second objective: ", second_objective)
    
    # remove the files generated from training
    os.remove(ret_pc_path)
    os.remove(model_path)
    os.remove(new_gt_pc_path)
    os.remove(os.path.join(OUTPUT_DIR, f"meta_{identifier}.ply"))
    
    return first_objective, second_objective
    
if __name__ == "__main__":
    # Test a single run
    identifier = run_single_meta_iteration(
        "display",
        num_meta_loops=10,
        num_inner_iterations=500,
        meta_lr=5e-2,
        inner_lr=1e-2,
        log2_hashmap_size=16,
        per_level_scale=1.38191,
        n_neurons=32,
        n_hidden_layers=2
    )
    
    resutls = evaluate_run(
        "display",
        identifier=identifier,
        inner_lr=1e-2,
        log2_hashmap_size=16,
        per_level_scale=1.38191,
        n_neurons=32,
        n_hidden_layers=2
    )
    
    print("Mean accuracy and completion: ", resutls[0])
    print("Model size: ", resutls[1])