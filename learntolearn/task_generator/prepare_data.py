import os
import sys
import json
import shutil
import random
import numpy as np
from math import ceil
import sapien.core as sapien

from helpers import *
from constants import *

"""Usage: python prepare_data.py <category_name>"""
IS_THIRD_PARTY = False

def main():
    # get the category name from the command line
    category_name = sys.argv[1]
    identifier = ""
    if len(sys.argv) > 2:
        identifier = sys.argv[2]
        
    category_id = CATEGORY_IDS[category_name]
    
    # get a list of all the objects in the category
    root_data_dir = "sapiens_data/{}/".format(category_name)
    if IS_TEST:
        root_data_dir = os.path.join(root_data_dir, "test/")
    object_dirs = [root_data_dir + d for d in os.listdir(root_data_dir) if os.path.isdir(root_data_dir + d)
                                                                        and not d.startswith("test")]
    
    # distribute the number of poses across the objects - ceil division
    num_objects = len(object_dirs)
    poses_per_object = ceil(NUM_POSES / num_objects)
    
    if THIRD_PARTY_ONLY:
        object_dirs = [directory for directory in object_dirs if "object.obj" in os.listdir(directory)]
    
    # create the output directory
    output_path = f"output/{category_name}/"
    if IS_TEST:
        output_path += "test/"
    if identifier:
        output_path += f"{identifier}/"
    if IS_TEST:
        object_dirs = random.choices(object_dirs, k=1)
        poses_per_object = NUM_TEST_POSES
    elif CHOOSE_ONE_RANDOM:
        object_dirs = random.choices(object_dirs, k=1)
        poses_per_object = NUM_POSES
    
    if len(object_dirs) == 0:
        print("No objects found with the constraints")
        return
    
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)
    for dir_name in ("rgb", "depth", "instance", "obj_offline", "models"):
        os.makedirs(os.path.join(output_path, dir_name), exist_ok=True)
    
    # the file handles - just to improve readability
    obj_meta_f, camera_poses_f, img_txt_f = None, None, None
    
    # time references for the images
    start_time, time_step = 0.00, 0.05
    
    # track the overall size of the objects
    overall = np.zeros(3)
    centers = np.zeros(3)
    
    # start generating the images
    for obj_idx, object_dir in enumerate(object_dirs):
        # create the simulator 
        print("Object: ", object_dir)
        engine = sapien.Engine()
        renderer = sapien.SapienRenderer(offscreen_only=True)
        engine.set_renderer(renderer)
        scene = engine.create_scene()
        scene.set_timestep(1 / 100.0)
        loader = None
        asset = None
        
        # the object pose
        Tso = np.eye(4)
        Two = Tws @ Tso
        
        # get the bounding box of the object to figure out the scaling factor
        with open(object_dir + "/bounding_box.json", "r") as f:
            bbox = json.load(f)
        
        # check if object is from the Sapiens dataset or third-party
        if os.path.exists(object_dir + "/mobility.urdf"):
            IS_THIRD_PARTY = False
        else:
            IS_THIRD_PARTY = True
        
        # bounding box points are in the box coordinate system - p_b, need them in the object coordinate system - p_o
        # p_o = Tob @ p_b
        if IS_THIRD_PARTY:
            min_bbox_point = np.array(bbox["min"])
            max_bbox_point = np.array(bbox["max"])
        else:
            min_bbox_point = np.linalg.inv(Tbo) @ np.vstack([np.array(bbox["min"]).reshape(3, 1), [1]])
            max_bbox_point = np.linalg.inv(Tbo) @ np.vstack([np.array(bbox["max"]).reshape(3, 1), [1]])
        
        # scale the object to the mean size - object pose estimation code depends on the mean
        scale = 0.25
        if MEAN_SIZES.get(category_name) is not None:
            lengths = np.abs(max_bbox_point[:3] - min_bbox_point[:3])
            scales = np.array([MEAN_SIZES[category_name][i] / lengths[i] for i in range(3)])
            scale = np.min(scales)
            if CHOOSE_ONE_RANDOM or IS_TEST:
                # add a random factor to the scale
                scale *= random.uniform(0.9, 1.1)
        
        # load the object
        if IS_THIRD_PARTY:
            loader = scene.create_actor_builder()
            loader.add_collision_from_file(object_dir + "/object.obj", scale=np.array([scale]*3))
            loader.add_visual_from_file(object_dir + "/object.obj", scale=np.array([scale]*3))
            asset = loader.build_static()
            asset.set_pose(sapien.Pose.from_transformation_matrix(Tso))
            
            # save the 3D model for evaluation
            shutil.copy(object_dir + "/object.obj", os.path.join(output_path, f"{obj_idx}.obj"))
        else:
            loader = scene.create_urdf_loader()
            loader.fix_root_link = True
            loader.scale = scale
            asset = loader.load_file_as_articulation_builder(object_dir + "/mobility.urdf")
            asset = asset.build_kinematic()
            asset.set_pose(sapien.Pose.from_transformation_matrix(Tso))
            
        # inflate the bounding box to make sure the object is not cut off
        min_bbox_point = min_bbox_point[:3].flatten() * scale * (1 + BBOX3D_PADDING)
        max_bbox_point = max_bbox_point[:3].flatten() * scale * (1 + BBOX3D_PADDING)
        
        # add 1cm padding to the bounding box
        min_bbox_point += min_bbox_point/np.abs(min_bbox_point) * 0.005
        max_bbox_point += max_bbox_point/np.abs(max_bbox_point) * 0.005
        center = (max_bbox_point + min_bbox_point) / 2
            
        overall += np.abs(max_bbox_point - min_bbox_point)
        centers += center
        print("Min bbox point: ", min_bbox_point)
        print("Max bbox point: ", max_bbox_point)
        print("Scaling factor: ", scale)
        
        Tso[:3, -1] = center
        Two = Tws @ Tso
        min_bbox_point = min_bbox_point - center
        max_bbox_point = max_bbox_point - center
        TWO = Two.copy()
        
        if not CHOOSE_ONE_RANDOM and not IS_TEST:
            mean_size = MEAN_SIZES[category_name]
            bbox = {
                "min": np.array([-mean_size[0]/2-0.005, -mean_size[1]/2-0.005, -mean_size[2]/2-0.005]),
                "max": np.array([mean_size[0]/2+0.005, mean_size[1]/2+0.005, mean_size[2]/2+0.005])
            }
            # add additional padding to the bounding box
            min_bbox_point = np.array(bbox["min"]) * (1 + BBOX3D_PADDING)
            max_bbox_point = np.array(bbox["max"]) * (1 + BBOX3D_PADDING)
            
            min_bbox_point -= np.array(EXTRA_PADDING[category_name])
            max_bbox_point += np.array(EXTRA_PADDING[category_name])            
            
            TSO = np.eye(4)
            TWO = Tws @ TSO

        # set some lighting
        # [TODO] - add random lighting
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

        # set the camera
        camera = scene.add_camera(
            name="camera",
            width=WIDTH,
            height=HEIGHT,
            fovy=FOVY,
            near=NEAR,
            far=FAR,
        )

        # object pose computation
        if obj_idx == 0:
            tx, ty, tz = TWO[:3, 3]
            qx, qy, qz, qw = rotation_matrix_to_quaternion(TWO[:3, :3])
            a1, a2, a3 = -np.abs(min_bbox_point)
            a4, a5, a6 = np.abs(max_bbox_point)
            
            # create the necessary files 
            obj_meta_f = open(os.path.join(output_path, "obj_offline", "0.txt"), "w")
            camera_poses_f = open(os.path.join(output_path, "groundtruth.txt"), "w")
            img_txt_f = open(os.path.join(output_path, "img.txt"), "w")
            
            # write the header
            camera_poses_f.write("#timestamp tx ty tz qx qy qz qw\n")
            img_txt_f.write("#timestamp filename -> rgb/ depth/ instance/\n")
            obj_meta_f.write("#category_id tx ty tz qx qy qz qw a1 a2 a3 a4 a5 a6\n")
            obj_meta_f.write(f"{category_id} {tx} {ty} {tz} {qx} {qy} {qz} {qw} {a1} {a2} {a3} {a4} {a5} {a6}\n")
            
            scale_file = open(os.path.join(output_path, "obj_offline/scale.txt"), "w")
            scale_file.write(f"{scale}\n")
            scale_file.close()
            
            # save the camera intrinsics
            with open(os.path.join(output_path, "intrinsics.json"), "w") as f:
                json.dump({
                    "width": WIDTH,
                    "height": HEIGHT,
                    "fx": camera.fx,
                    "fy": camera.fy,
                    "cx": camera.cx,
                    "cy": camera.cy,
                    "near": NEAR,
                    "far": FAR
                }, f, indent=4)
        
        # start generating the images
        prev_incr = obj_idx*poses_per_object*time_step
        for i in range(poses_per_object):
            current_time = "{:.6f}".format(start_time + prev_incr + i*time_step)
            point = random_point_in_sphere(Tso[:3, -1],
                                           SAMPLING_RADIUS_RANGE,
                                           SAMPLING_THETA_RANGE,
                                           SAMPLING_PHI_RANGE)
            rgba_pil, depth_pil, seg_pil, bbox, Twc = render_img(point, Tso, scene, camera, category_id)   
            rgba_pil.save(os.path.join(output_path, "rgb", f"{current_time}.png"))
            depth_pil.save(os.path.join(output_path, "depth", f"{current_time}.png"))
            seg_pil.save(os.path.join(output_path, "instance", f"{current_time}.png"))

            # save the object meta data
            obj_meta_f.write(f"{current_time} {' '.join([str(x) for x in bbox])}\n")
            
            # convert the camera pose to tx ty tz qx qy qz qw
            tx, ty, tz = Twc[:3, 3]
            qx, qy, qz, qw = rotation_matrix_to_quaternion(Twc[:3, :3])
            camera_poses_f.write(f"{current_time} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
            
            # write the timestamp and the image file names
            img_txt_f.write(f"{current_time} {current_time}.png\n")
            
        # delete the simulator
        del camera, asset, loader, scene, renderer, engine
            
    # close the files
    obj_meta_f.close()
    camera_poses_f.close()
    img_txt_f.close()
    
    # copy the config.yaml 
    shutil.copy("configs/config.yaml", os.path.join(output_path, "config.yaml"))
    
    # print the overall size of the objects
    print("Overall size: ", overall/num_objects)
    print("Mean Center: ", centers/num_objects)
    
    
if __name__ == "__main__":
    main()