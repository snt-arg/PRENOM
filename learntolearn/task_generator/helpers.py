import math
import random
import numpy as np
from PIL import Image
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R
from constants import DEPTH_SCALE


# (world to sapien-world) - determined by visually comparing the object in the simulator and the world in the RO-MAP
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


def random_point_in_sphere(object_position, radius_range, theta_range=[0, 2*math.pi], phi_range=[0, math.pi]):
    """Generate a random point in a sphere around the object - uniformly distributed"""
    # Generate random spherical coordinates around the object
    r = random.uniform(*radius_range)
    theta = random.uniform(*theta_range)
    phi = random.uniform(*phi_range)
    
    # Convert spherical coordinates to cartesian coordinates
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    
    # Translate the point by the object's position
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


def get_depth(camera):
    """returns depth image from the camera - is in the OpenGL frame: p_c"""
    position = camera.get_float_texture('Position')  # [H, W, 4]
    depth = -position[..., 2]
    depth_image = (depth * DEPTH_SCALE).astype(np.uint16)
    depth_pil = Image.fromarray(depth_image)
    return depth_pil


def rotation_matrix_to_quaternion(rot_mat):
    # result is in qx, qy, qz, qw format
    rot = R.from_matrix(rot_mat)
    return rot.as_quat()


def get_bbox_from_mask(mask, half_padding=0.08):
    """returns a padded bounding box from the mask"""
    mask = np.array(mask)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # inflate the bbox, clip to image size, round to integer
    rmin = round(max(0, rmin - half_padding * (rmax - rmin)))
    rmax = round(min(mask.shape[0], rmax + half_padding * (rmax - rmin)))
    cmin = round(max(0, cmin - half_padding * (cmax - cmin)))
    cmax = round(min(mask.shape[1], cmax + half_padding * (cmax - cmin)))
    
    # # [DEBUG] - crop the image to the bounding box and visualize
    # import cv2
    # cropped_mask = mask[rmin:rmax, cmin:cmax]
    # cv2.imshow("cropped", cropped_mask.astype(np.uint8) * 255)
    # cv2.waitKey(200)

    # return in x, y, h, w format - because that is what RO-MAP expects
    # [TODO] - make RO-MAP accept x, y, w, h format
    return cmin, rmin, rmax - rmin, cmax - cmin


def render_img(point, Tso, scene, camera, category_id):
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
    rgba_img[mask == 0] = 0
    
    # convert the images to PIL format
    rgba_pil = Image.fromarray(rgba_img[:, :, : 3], 'RGB')
    depth_pil = get_depth(camera)
    seg_pil = Image.fromarray(mask.astype(np.uint8) * category_id)
    
    return rgba_pil, depth_pil, seg_pil, get_bbox_from_mask(mask), Twc


def generate_pointcloud(camera, Tso):
    """Generate a point cloud from the camera"""
    position = camera.get_float_texture('Position')  # [H, W, 4]
    points_opengl = position[..., :3][position[..., 3] < 1]
    model_matrix = camera.get_model_matrix()
    points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3] - Tso[:3, -1]
    return points_world

def sample_points(pcl, n_pts):
    """ Down sample the point cloud using farthest point sampling.

    Args:
        pcl (torch tensor or numpy array):  NumPoints x 3
        num (int): target point number
    """
    total_pts_num = pcl.shape[0]
    if total_pts_num < n_pts:
        pcl = np.concatenate([np.tile(pcl, (n_pts // total_pts_num, 1)), pcl[:n_pts % total_pts_num]], axis=0)
    elif total_pts_num > n_pts:
        ids = np.random.permutation(total_pts_num)[:n_pts]
        pcl = pcl[ids]
    return pcl

def generate_random_model_id(k=10):
    """Generate a random model id for the object with character length k"""
    return ''.join(random.choices('0123456789abcdefghijklmnopqrstuvwxyzQWERTYUIOPLKJHGFDSAZXCVBNM', k=k))