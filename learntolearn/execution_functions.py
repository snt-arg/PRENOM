import subprocess
import json
import numpy as np
import os
import time
import shutil
import trimesh
from pyntcloud import PyntCloud
import random
import sys

from reconstruction_evaluation import accuracy, completion
from config import *


def single_train_call(
    base_path,
    system_path,
    data_dir
):
    # call the object training script and wait for it to finish
    command = f"cd {SCRIPT_DIR} && __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./build/OfflineNeRF {base_path} {system_path} {data_dir}"
    object_training = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    object_training.wait()
    out, _ = object_training.communicate()
    return out


def run_single_meta_iteration(
    category,
    num_meta_loops,
    num_inner_iterations,
    meta_lr,
    meta_lr_decay,
    inner_lr,
    log2_hashmap_size,
    per_level_scale,
    n_neurons,
    n_hidden_layers,
    density_lambda,
    depth_lambda,
    depth_prescale,
    save_prior = False
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
    
    # output directory based on category
    output_dir = os.path.join(OUTPUT_DIR, category, "") # need a trailing slash
    os.makedirs(output_dir, exist_ok=True)
    
    # create an identifier for this run
    identifier = np.random.randint(0, 100000000)
    
    # get all the available data within the category
    # [TODO] - use an Object-Oriented approach to get the entire dataset only once
    data = os.listdir(os.path.join(TASKS_DIR, category, "train"))
    
    # create a base json file
    with open(os.path.join(SCRIPT_DIR, 'Core/configs/base.json'), 'r') as file:
        base = json.load(file)
        
    new_base = base.copy()
    new_base["optimizer"]["nested"]["nested"]["learning_rate"] = inner_lr
    new_base["meta_optimizer"]["nested"]["nested"]["learning_rate"] = meta_lr
    new_base["encoding"]["log2_hashmap_size"] = log2_hashmap_size
    new_base["encoding"]["per_level_scale"] = per_level_scale
    new_base["network"]["n_neurons"] = n_neurons
    new_base["network"]["n_hidden_layers"] = n_hidden_layers
    new_base["lambdas"]["density"] = density_lambda
    new_base["lambdas"]["depth"] = depth_lambda
    new_base["misc"]["depth_prescale"] = depth_prescale
    
    # save the new json file
    base_path = os.path.join(output_dir, f'base_{identifier}.json')
    with open(base_path, 'w') as file:
        json.dump(new_base, file)
        
    # create a system json file
    new_system = {
        "use_depth": True,
        "do_meta": True,
        
        "object_indices": [0],
        
        "save_model": True,
        "save_identifier": identifier,
        
        "output_dir": output_dir,
        
        "load_model": False,
        "load_path": "",
        
        "visualize": False,
        
        "n_iters_per_step": num_inner_iterations,
        "n_steps": 1,
        
        "meta_n_iters_per_step": num_inner_iterations,
        "meta_n_steps": 1,
        "meta_n_loops": 1
    }
    
    system_path = os.path.join(output_dir, f'system_{identifier}.json')
    with open(system_path, 'w') as file:
        json.dump(new_system, file)
    
    # call the single call function to get a starting model
    data_dir = random.choice(data)
    data_dir = os.path.join(TASKS_DIR, category, "train", data_dir)
    single_train_call(base_path, system_path, data_dir)
    
    # now change the new_system to load the generated model
    new_system["load_model"] = True
    new_system["load_path"] = os.path.join(output_dir, f"meta_{identifier}.json")
    with open(system_path, 'w') as file:
        json.dump(new_system, file)
        
    # run the meta loop
    for i in range(num_meta_loops - 1):
        data_dir = random.choice(data)
        data_dir = os.path.join(TASKS_DIR, category, "train", data_dir)

        # update the meta learning rate        
        meta_lr = meta_lr * meta_lr_decay
        new_base["meta_optimizer"]["nested"]["nested"]["learning_rate"] = meta_lr
        with open(base_path, 'w') as file:
            json.dump(new_base, file, indent=4)
        
        # the training
        print(f"Meta training iter {i} with data dir: {data_dir}")
        single_train_call(base_path, system_path, data_dir)
        
        # time to save the prior model if requested
        if i == num_meta_loops - 2 and save_prior:
            # make the prior folder
            prior_dir = os.path.join("priors", category)
            os.makedirs(prior_dir, exist_ok=True)
            
            # copy the model and json files
            shutil.copyfile(os.path.join(output_dir, f"meta_{identifier}.ply"), os.path.join(prior_dir, "model.ply"))
            shutil.copyfile(os.path.join(output_dir, f"{identifier}_density.ply"), os.path.join(prior_dir, "density.ply"))
            shutil.copyfile(os.path.join(output_dir, f"meta_{identifier}.json"), os.path.join(prior_dir, "weights.json"))
            shutil.copyfile(base_path, os.path.join(prior_dir, "network.json"))
            
            # need to normalize the model.ply file
            with open(f"{data_dir}/obj_offline/0.txt" , "r") as f:
                # ignore the first line (comment) and the first element of the second line (category id)
                lines = f.readlines()[1:]
                _, _, _, _, _, _, _, a1, a2, a3, a4, a5, a6 = [float(x) for x in lines[0].split()[1:]]
            
            # the extent of the object
            bbox_min = np.array([a1, a2, a3])
            bbox_max = np.array([a4, a5, a6])
            
            # normalize the pointcloud and save it
            pc = trimesh.load(os.path.join(output_dir, f"meta_{identifier}.ply"))
            pc.vertices = (pc.vertices - bbox_min) / (bbox_max - bbox_min)
            print(pc.vertices.min(axis=0), pc.vertices.max(axis=0))
            pc.export(os.path.join(prior_dir, "model.ply"))
            
    # delete the json and model files
    os.remove(base_path)
    os.remove(system_path)
    os.remove(os.path.join(output_dir, f"meta_{identifier}.ply"))
    os.remove(os.path.join(output_dir, f"{identifier}_density.ply"))
    os.remove(os.path.join(output_dir, f"meta_{identifier}.json"))

    return identifier
  
def evaluate_run(
    category,
    identifier,
    inner_lr,
    log2_hashmap_size,
    per_level_scale,
    n_neurons,
    n_hidden_layers,
    density_lambda,
    depth_lambda,
    depth_prescale,
    load_meta_model = True,
    inner_loops_to_test = [80, 160, 240],
    use_depths = [True, False],
    evaluate_folder = "test"
):
    output_dir = os.path.join(OUTPUT_DIR, category, "") # need a trailing slash
    os.makedirs(output_dir, exist_ok=True)

    # get all the test sets
    test_dir = os.path.join(TASKS_DIR, category, evaluate_folder)
    test_sets = os.listdir(test_dir)
    print(test_sets)
    
    # set a new base config
    with open(f'{SCRIPT_DIR}/Core/configs/base.json', 'r') as file:
        base = json.load(file)
        
    new_base = base.copy()
    new_base["optimizer"]["nested"]["nested"]["learning_rate"] = inner_lr
    new_base["encoding"]["log2_hashmap_size"] = log2_hashmap_size
    new_base["encoding"]["per_level_scale"] = per_level_scale
    new_base["network"]["n_neurons"] = n_neurons
    new_base["network"]["n_hidden_layers"] = n_hidden_layers
    new_base["lambdas"]["density"] = density_lambda
    new_base["lambdas"]["depth"] = depth_lambda
    new_base["misc"]["depth_prescale"] = depth_prescale
    
    # save the new json file
    base_path = os.path.join(output_dir, f'base_{identifier}.json')
    with open(base_path, 'w') as file:
        json.dump(new_base, file)
    
    # set the system config
    model_path = os.path.join(output_dir, f"{identifier}_0.json")
    meta_model_path = os.path.join(output_dir, f"meta_{identifier}.json")
    new_system = {
        "do_meta": False,
        
        "object_indices": [0],
        
        "save_model": True,
        "save_identifier": identifier,
        
        "output_dir": output_dir,
        
        "load_model": load_meta_model,
        "load_path": meta_model_path,
        
        "visualize": False,
        
        "n_iters_per_step": 50,
        
        "meta_n_iters_per_step": 1,
        "meta_n_steps": 1,
        "meta_n_loops": 1
    }
    system_path = os.path.join(output_dir, f'system_{identifier}.json')
    with open(system_path, 'w') as file:
        json.dump(new_system, file)
  
    accuracies = []
    completions = []
  
    total_time = 0
    ret_pc_path = os.path.join(output_dir, f"{identifier}_0.ply")
    new_gt_pc_path = os.path.join(output_dir, f"{identifier}_gt.ply")
    for depth_flag in use_depths:
        new_system["use_depth"] = depth_flag
        depth_accuracies = []
        depth_completions = []
        
        for inner_loops in inner_loops_to_test:
            new_system["n_steps"] = int(inner_loops/50)
            with open(system_path, 'w') as file:
                json.dump(new_system, file)
            
            for test_set in test_sets:
                data_dir = os.path.join(test_dir, test_set)
                out = single_train_call(base_path, system_path, data_dir)
                # scrape the time from the output
                out = out.decode("utf-8")
                lines = out.split("\n")
                for line in lines:
                    if "train_time: " in line:
                        total_time += float(line.partition("train_time: ")[2].partition(" ")[0])
                print("Total time: ", total_time)
                sys.stdout.flush()
                
                # get the scale of the object
                with open(os.path.join(test_dir, test_set, "obj_offline/scale.txt"), 'r') as file:
                    line = file.readlines()[0]
                    scale = float(line.strip())
                
                # get the resulting mesh - sample fixed number of points from both meshes
                ret_pc = PyntCloud.from_file(ret_pc_path)
                
                # if no vertices are generated, skip the evaluation
                if len(ret_pc.points) == 0:
                    depth_accuracies.append(np.nan)
                    depth_completions.append(np.nan)
                    continue
                
                ret_pc = ret_pc.get_sample("mesh_random", n=NUM_SAMPLED_POINTS, rgb=False, normals=False)
                ret_pc = trimesh.Trimesh(vertices=ret_pc[["x", "y", "z"]].values)
                
                # the ground truth mesh
                gt_pc = os.path.join(test_dir, test_set, "0.obj")
                gt_pc = trimesh.load(gt_pc, force='mesh')
                gt_pc.vertices = gt_pc.vertices * scale
                gt_pc.export(new_gt_pc_path)
                gt_pc = PyntCloud.from_file(new_gt_pc_path)
                gt_pc = gt_pc.get_sample("mesh_random", n=NUM_SAMPLED_POINTS, rgb=False, normals=False)
                gt_pc = trimesh.Trimesh(vertices=gt_pc[["x", "y", "z"]].values)
                
                # evaluate the meshes
                accuracy_metric = accuracy(gt_pc.vertices, ret_pc.vertices)
                completion_metric = completion(gt_pc.vertices, ret_pc.vertices)
                depth_accuracies.append(accuracy_metric)
                depth_completions.append(completion_metric)
        
        print("Depth flag: ", depth_flag)
        print("Mean accuracy: ", np.mean(depth_accuracies))
        print("Mean completion: ", np.mean(depth_completions), "\n")
        
        accuracies.extend(depth_accuracies)
        completions.extend(depth_completions)
    
    print("Final results:")
    print("Mean accuracy: ", np.mean(accuracies))
    print("Mean completion: ", np.mean(completions))
    
    # replace nans in accuracies
    accuracies = [acc if np.isfinite(acc) else 2.5 for acc in accuracies]
    completions = [comp if np.isfinite(comp) else 2.5 for comp in completions]
    
    # average chamfer distance in mm
    first_objective = (np.mean(accuracies) + np.mean(completions)) * 500
    print("First objective: ", first_objective)
  
    # remove the json files
    os.remove(base_path)
    os.remove(system_path)
  
    # second objective is the size of the saved model and training time
    model_size = 0
    if load_meta_model:
        model_size = os.path.getsize(meta_model_path)
    else:
        model_size = os.path.getsize(model_path)
    model_size = model_size / (1024 * 1024) # in MB
    print("Model size: ", model_size)
    
    total_iterations = np.sum(inner_loops_to_test) * len(use_depths) * len(test_sets)
    time_per_iteration = total_time / total_iterations # in ms
    print("Time per iteration: ", time_per_iteration)
    
    second_objective = LAMBDA_MODEL_SIZE * model_size + LAMBDA_TRAIN_TIME * time_per_iteration
    print("Second objective: ", second_objective)
    sys.stdout.flush()
    
    # remove the files generated from training
    try:
        os.remove(new_gt_pc_path)
    except OSError:
        print("None of the runs succeeded :((")
        pass
    os.remove(ret_pc_path)
    os.remove(model_path)
    if load_meta_model:
        os.remove(meta_model_path)
        os.remove(os.path.join(output_dir, f"meta_{identifier}.ply"))
    
    return first_objective, second_objective
    