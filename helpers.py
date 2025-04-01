import math
import random
import numpy as np
from PIL import Image
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R


# (world to sapien-world) - determined by visually comparing the object in the simulator and the world in PRENOM
Tws = np.array([
    [0, -1, 0, 0], 
    [0, 0, -1, 0], 
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])

# (bounding_box.json to object) - determined a bit empirically
Tbo = np.array([
    [0, -1, 0, 0], 
    [0, 0, 1, 0], 
    [-1, 0, 0, 0],
    [0, 0, 0, 1]
])


def get_spherical_trajectory(object_position, radius, min_theta=2*np.pi+np.pi/2, max_theta=+np.pi/2 ,min_phi=np.pi/4, max_phi=np.pi/8, millisecond = 0, total_time = 4000):
    """Generate a spherical trajectory around the object"""
    theta = (max_theta - min_theta) * millisecond/total_time + min_theta
    phi = 0
    if millisecond/total_time < 0.5:
        phi = 2*millisecond/total_time * (max_phi - min_phi) + min_phi
    else:
        phi = 2*(millisecond/total_time - 0.5) * (min_phi - max_phi) + max_phi
    
    return get_static_point(object_position, radius, theta, phi)


def get_static_point(object_position, radius, theta=0, phi=0):
    """Generate a static point around the object"""
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    # Translate the points by the object's position
    return np.array([x, y, z]) + object_position
    

def calculate_cam_ext(point, Tso):
    """returns cam_ext given camera position and Tso, assuming camera is looking at the object
    
    Reference: https://sapien.ucsd.edu/docs/2.2/tutorial/basic/hello_world.html"""
    cam_pos = np.array(point)
    forward = cam_pos - Tso[:3, -1]
    forward = -forward / np.linalg.norm(forward)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = np.array(point)
    return mat44


def calculate_cam_ext_gl(point, Tso):
    """returns Twc given camera position and Tso, assuming camera is looking at the object"""
    trans_gl = np.dot(Tws[:3, :3], point)
    Tgo = np.dot(Tws[:3, :3], Tso[:3, -1])
    forward = (trans_gl - Tgo)
    forward = -forward / np.linalg.norm(forward)
    right = np.cross([0, 1, 0], forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([right, up, forward], axis=1)
    mat44[:3, 3] = np.array(trans_gl)
    return mat44


def rotation_matrix_to_quaternion(rot_mat):
    # result is in qx, qy, qz, qw format
    rot = R.from_matrix(rot_mat)
    return rot.as_quat()


def render_img(point, Tso, scene, camera):
    """Renders an image from the camera at the given point"""
    # calculate the camera pose as expected by sapien
    mat44 = calculate_cam_ext(point, Tso)
    camera.set_pose(sapien.Pose.from_transformation_matrix(mat44))

    # render the image
    scene.step()
    scene.update_render()
    camera.take_picture()
    
    # calculate the camera pose as expected by OpenGL for NeRF 
    Twc = calculate_cam_ext_gl(point, Tso)

    # get the RGBA image and the segmentation mask
    rgba = camera.get_float_texture('Color')  # [H, W, 4]        
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
    mask = seg_labels.sum(axis=-1)
    mask[mask>0] = 1
    rgba_img[:, :, -1] = rgba_img[:, :, -1] * mask
    
    # put a black background - could be a random color as well
    rgba_img[mask == 0] = 255
    
    # convert the images to PIL format
    rgba_pil = Image.fromarray(rgba_img[:, :, : 3], 'RGB')
    
    return rgba_pil