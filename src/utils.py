import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import quat_to_angle_axis, quat_from_coeffs
import quaternion
from scipy.spatial import KDTree
import logging
import open3d as o3d
def generate_candidate_viewpoints(bbox_center, radius, pts, num_points=20):
    """Generate candidate viewpoints around the target bounding box center on a circle."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    viewpoints = []
    for angle in angles:
        x = bbox_center[0] + radius * np.cos(angle)### x0 + r*cos(theta)
        y = bbox_center[2] + radius * np.sin(angle)### y0 + r*sin(theta)
        z = pts[1]  # Z remains constant
        viewpoints.append(np.array([x, z, y]))
    return np.array(viewpoints)



def is_point_visible(viewpoint, target_point, scene_points_tree, threshold=0.05):
    """Check if a target point is visible from a viewpoint considering scene occlusion."""
    direction = target_point - viewpoint
    view_distance = np.linalg.norm(direction)
    direction /= view_distance

    num_samples = min(1000, int(view_distance / threshold) + 1)
    sample_points = np.array([
        viewpoint + t * direction 
        for t in np.linspace(3 * threshold, view_distance - 3 * threshold, num_samples)
    ])
    
    # Batch query nearest neighbors for all sampled points
    distances, indices = scene_points_tree.query(sample_points, k=1)
    # Check if any sampled point is too close to scene points (occlusion)
    return not np.any(distances < threshold)

def compute_visibility(viewpoint, target_points, scene_points_tree):
    """Compute visibility of target points from a given viewpoint considering scene occlusion."""
    visible_count = 0

    for target_point in target_points:
        visible = is_point_visible(viewpoint, target_point, scene_points_tree)
        if visible:
            visible_count += 1

    return visible_count / target_points.shape[0]


def Visibility_based_Viewpoint_Decision(target_points, scene_points, pts, tsdf_planner, radius_factor):
    target_points = target_points[np.random.choice(target_points.shape[0], min(1000, target_points.shape[0]), replace=False)]
    scene_points_tree = KDTree(scene_points)
    bbox_center = target_points.mean(axis=0)
    best_visibility = 0
    best_viewpoint = None
    candidate_viewpoints = generate_candidate_viewpoints(bbox_center, radius_factor, pts)
    filtered_viewpoints = tsdf_planner.mask_true_point(candidate_viewpoints)
    for vp in filtered_viewpoints:### filtered_viewpoints: candidates of best_viewpoint
        vp[1] += 1.5 #camera height
        visibility_score = compute_visibility(vp, target_points, scene_points_tree)
        vp[1] -= 1.5
        if visibility_score > best_visibility:
            best_visibility = visibility_score
            best_viewpoint = vp
    if best_viewpoint is None:
        near_viewpoints = tsdf_planner.get_near_true_point(candidate_viewpoints)
        for vp in near_viewpoints:
            vp[1] += 1.5
            visibility_score = compute_visibility(vp, target_points, scene_points_tree)
            vp[1] -= 1.5
            if visibility_score > best_visibility: ###update: vp is best view point
                best_visibility = visibility_score
                best_viewpoint = vp
    print("Best viewpoint:", best_viewpoint)
    return best_viewpoint
    """Calculate the best viewpoint from a set of candidate viewpoints."""
    # Prepare data

def resize_image(image, target_h, target_w):
    # image: np.array, h, w, c
    image = Image.fromarray(image)
    image = image.resize((target_w, target_h))
    return np.array(image)


def find_center_in_room(centers, confidences, xyxy, class_ids, rooms):
    if len(confidences) > 0:
        sorted_indices = np.argsort(confidences)[::-1]
        class_ids = class_ids[sorted_indices]
        confidences = confidences[sorted_indices]
        xyxy = xyxy[sorted_indices]
    room_label = []
    room_conf = []
    for center in centers:   
        find_room = False
        x, y = center
        for idx in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[idx]
            if x1 <= x <= x2 and y1 <= y <= y2:
                room_label.append(rooms[class_ids[idx]])
                room_conf.append(confidences[idx])
                find_room = True
                break
        if not find_room:
            room_label.append('unknown')
    return room_label, room_conf

def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, "RGBA image has 4 channels."

    rgb = np.zeros((row, col, 3), dtype="float32")
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype="float32") / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype="uint8")


def get_pts_angle_aeqa(init_pts, init_quat):
    pts = np.asarray(init_pts)

    init_quat = quaternion.quaternion(*init_quat)
    angle, axis = quat_to_angle_axis(init_quat)
    angle = angle * axis[1] / np.abs(axis[1])

    return pts, angle


def get_pts_angle_goatbench(init_pos, init_rot):
    pts = np.asarray(init_pos)

    init_quat = quat_from_coeffs(init_rot)
    angle, axis = quat_to_angle_axis(init_quat)
    angle = angle * axis[1] / np.abs(axis[1])

    return pts, angle

def calc_agent_subtask_distance(curr_pts, viewpoints, pathfinder):
    # calculate the distance to the nearest view point
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = curr_pts
    path.requested_ends = viewpoints
    # np.save(f'/hsun/0625/3D-Mem/vis/pos_{curr_pts[1]}.npy', np.array(viewpoints))
    found_path = pathfinder.find_path(path)
    if found_path:
        distance = path.geodesic_distance
    else:
        distance = 10.0
    return distance