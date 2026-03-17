import logging
from typing import Tuple, Optional, Union
import random
import numpy as np

from src.explore_utils import task_check, explore_two_step
from src.utils import Visibility_based_Viewpoint_Decision
from src.tsdf_planner import TSDFPlanner, Frontier
from src.multimodal_3d_scene_graph import Scene
from src.conceptgraph.slam.utils import (
    get_bounding_box,
    init_process_pcd,
    detections_to_obj_pcd_and_bbox,
)
def random_frontier_choice(tsdf_planner: TSDFPlanner, n_filtered_snapshots):
    """
    Choose a random frontier from the TSDF planner.
    """
    if len(tsdf_planner.frontiers) == 0:
        logging.error("No frontiers available, returning None.")
        return None
    idx = random.randint(0, len(tsdf_planner.frontiers)-1)
    random_frontier = tsdf_planner.frontiers[idx]
    logging.info(f"Randomly chosen frontier at {random_frontier.position}")
    return "frontier", random_frontier, n_filtered_snapshots, idx


def get_aabb_corner_points(aabb):
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    
    return np.array([
        [min_bound[0], min_bound[1], (max_bound[2] + min_bound[2]) / 2],
        [max_bound[0], min_bound[1], (max_bound[2] + min_bound[2]) / 2],
        [(max_bound[0] + min_bound[0]) / 2, min_bound[1], min_bound[2]], 
        [(max_bound[0] + min_bound[0]) / 2, min_bound[1], max_bound[2]], 
    ])

def select_navigation_corner(aabb, selection_strategy="closest_to_robot", robot_position=None):
    """
    Select a bounding box corner point as the navigation target.
    
    Parameters:
    aabb: Open3D Axis-Aligned Bounding Box object
    selection_strategy: selection strategy ("closest_to_robot", "lowest", "front_center")
    robot_position: (3,) array representing the robot's current position [x, y, z]
                    (only required if the strategy depends on robot position)
    
    Returns:
    target: (3,) array containing the coordinates of the selected corner point
    """
    # Retrieve all corner points
    corners = get_aabb_corner_points(aabb)
    
    if selection_strategy == "closest_to_robot" and robot_position is not None:
        # Select the corner point closest to the robot (XY-plane distance)
        distances = np.linalg.norm(corners[:, [0, 2]] - robot_position[[0, 2]], axis=1)
        return corners[np.argmin(distances)]
    
    elif selection_strategy == "lowest":
        # Select the corner point with the lowest height (assuming the object is grounded)
        return corners[np.argmin(corners[:, 1] - robot_position[1])]
    
    elif selection_strategy == "front_center" and robot_position is not None:
        # Select the middle point of the face oriented towards the robot
        
        # 1. Determine the face facing the robot
        center = aabb.get_center()
        to_robot = robot_position - center
        to_robot[2] = 0  # Ignore height difference
        to_robot /= np.linalg.norm(to_robot)
        
        # 2. Compute center points of each face
        face_centers = [
            np.mean(corners[[0, 1, 2, 3]], axis=0),  # front face (min x)
            np.mean(corners[[4, 5, 6, 7]], axis=0),  # back face (max x)
            np.mean(corners[[0, 1, 4, 5]], axis=0),  # left face (min y)
            np.mean(corners[[2, 3, 6, 7]], axis=0)   # right face (max y)
        ]
        
        # 3. Specify directions for each face
        face_directions = [
            np.array([-1, 0, 0]),  # front face facing +X
            np.array([1, 0, 0]),   # back face facing −X
            np.array([0, -1, 0]),  # left face facing +Y
            np.array([0, 1, 0])    # right face facing −Y
        ]
        
        # Determine the face most oriented towards the robot by dot product
        dot_products = [np.dot(dir, to_robot) for dir in face_directions]
        selected_face = np.argmax(dot_products)
        
        # 4. Return the lowest corner point of the selected face (assuming floor-level navigation)
        face_corners = {
            0: [0, 1, 2, 3],  # front face
            1: [4, 5, 6, 7],  # back face
            2: [0, 1, 4, 5],  # left face
            3: [2, 3, 6, 7]   # right face
        }[selected_face]
        
        face_points = corners[face_corners]
        return face_points[np.argmin(face_points[:, 2])]
    
    else:
        # Default: select the lowest-height corner point
        return corners[np.argmin(corners[:, 2])]

def query_vlm_for_response(
    subtask_metadata: dict,
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    rgb_egocentric_views: list,
    cfg,
    pts = None,
    verbose: bool = False,
):
    # prepare input for vlm
    step_dict = {}

    # prepare object and image context
    object_id_to_name = {
        obj_id: obj["class_name"] for obj_id, obj in scene.objects.items()
    }
    object_id_to_room = {
        obj_id: [obj["room_label"],obj["room_conf"]] for obj_id, obj in scene.objects.items()
    }
    step_dict["obj_map"] = object_id_to_name

    step_dict["objects"] = scene.objects
    step_dict["all_imgs"] = scene.all_observations
    step_dict["edges"] = scene.edges
    step_dict["prompt_h"] = cfg.prompt_h
    step_dict["prompt_w"] = cfg.prompt_w
    step_dict["use_full_obj_list"] = cfg.use_full_obj_list

    # prepare frontier
    step_dict["frontier_imgs"] = [
        frontier.feature for frontier in tsdf_planner.frontiers
    ]

    # prepare egocentric views
    if cfg.egocentric_views:
        step_dict["egocentric_views"] = rgb_egocentric_views
        step_dict["use_egocentric_views"] = True

    # prepare other metadata
    step_dict["question"] = subtask_metadata["question"]
    step_dict["task_type"] = subtask_metadata["task_type"]
    step_dict["class"] = subtask_metadata["class"]
    step_dict["image"] = subtask_metadata["image"]
    step_dict["CLR"] = subtask_metadata['CLR']
    step_dict["object_id_to_room"] = object_id_to_room
    step_dict['image_to_edges'] = scene.img_to_edge
    # query vlm
    (
        outputs,
        image_map_reverse,
        reason,
        n_filtered_snapshots,
    ) = explore_two_step(step_dict, cfg, verbose=verbose)
    if outputs is None:
        logging.error(f"explore_step failed and returned None, Choose a random frontier instead!")
        return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
    logging.info(f"Response: [{outputs}]\nReason: [{reason}]")
    
    # parse returned results
    try:
        target_type, target_index = outputs.split(",")[0].strip().split(" ")
        logging.info(f"Prediction: {target_type}, {target_index}")
    except:
        logging.info(f"Wrong output format, Choose a random frontier instead!")
        return random_frontier_choice(tsdf_planner, n_filtered_snapshots)

    if target_type not in ["image", "frontier", "object"]:
        logging.info(f"Wrong target type: {target_type}, Choose a random frontier instead!")
        return random_frontier_choice(tsdf_planner, n_filtered_snapshots)


    if target_type == "image":
        #Implementation of AVU.
        #We update only the words we consider to be the target, 
        #so we perform re-perception directly and select it as the answer.
        if int(target_index) >= 0 and int(target_index) < len(image_map_reverse):
            target_index = image_map_reverse[int(target_index)]
        else:
            view_idx = int(target_index) - len(image_map_reverse)
            global_step = scene.global_step_cnt
            target_index = f"{global_step}-view_{view_idx}.png"
        target_image = scene.all_observations[target_index]
        try:
            object_class = outputs.split(",")[1].strip()
        except:
            logging.info(f"Wrong output format, Choose a random frontier instead!")
            return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
        scene.detection_model.set_classes([object_class])
        results = scene.detection_model.predict(
            target_image, conf=cfg.AVU_conf_threshold, verbose=False
        )
        
        scene.detection_model.set_classes(scene.obj_classes.get_classes_arr())
        if len(results) == 0 or len(results[0].boxes) == 0:
            logging.info(
                f"No objects detected in the predicted image: {target_index}, Choose a random frontier instead!"
            )
            return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
        
        confidences = results[0].boxes.conf.cpu().numpy()
        max_idx = confidences.argmax()
        max_confidence = confidences[max_idx: max_idx+1]
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        max_detection_class_ids = detection_class_ids[max_idx: max_idx+1]
        xyxy_tensor = results[0].boxes.xyxy[max_idx: max_idx+1, ...]
        sam_out = scene.sam_predictor.predict(
            target_image, bboxes=xyxy_tensor, verbose=False
        )
        masks_tensor = sam_out[0].masks.data
        masks_np = masks_tensor.cpu().numpy()
        obj_pcds_and_bboxes = detections_to_obj_pcd_and_bbox(
            depth_array=scene.all_depths[target_index],
            masks=masks_np,
            cam_K=scene.intrinsics[:3, :3],  # Camera intrinsics
            image_rgb=target_image,
            trans_pose=scene.all_cam_poses[target_index],
            min_points_threshold=scene.cfg_cg.min_points_threshold,
            spatial_sim_type=scene.cfg_cg.spatial_sim_type,
            obj_pcd_max_points=scene.cfg_cg.obj_pcd_max_points,
            device=scene.device,
        )
        
        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=scene.cfg_cg["downsample_voxel_size"],
                    dbscan_remove_noise=scene.cfg_cg["dbscan_remove_noise"],
                    dbscan_eps=scene.cfg_cg["dbscan_eps"],
                    dbscan_min_points=scene.cfg_cg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=scene.cfg_cg["spatial_sim_type"],
                    pcd=obj["pcd"],
                )
        try:
            a = []
            for idx in scene.objects.keys():
                a.append(scene.objects[idx]["pcd"].points)
            obj_pos = Visibility_based_Viewpoint_Decision(
                np.array(obj["pcd"].points),
                np.concatenate(a, axis=0),
                pts,
                tsdf_planner,
                cfg.dicision_radius,
            )
            if obj_pos is None:
                obj_pos = select_navigation_corner(aabb = obj["bbox"], robot_position = pts)
                logging.info(f"The index of target Image {target_index} : {obj_pos} (Closed Box Center, Confidence : {max_confidence})")
            else:
                logging.info(f"The index of target Image {target_index} : {obj_pos} (Visible Center, Confidence : {max_confidence})")
        except:
            logging.info(
                f"No Object Point Cloud in the predicted image: {target_index}, Choose a random frontier instead!"
            )
            return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
        # logging.info(f"The index of target Image {target_index} : {obj_pos} (Mask Center, Confidence : {max_confidence})")
        
        
        # a = []
        # for idx in scene.objects.keys():
        #     if idx != target_index:
        #         a.append(scene.objects[idx]["pcd"].points)
        # np.save(f'/hsun/0625/3D-Mem/vis/other_obj_{pts[1]}.npy', np.concatenate(a, axis=0))
        # np.save(f'/hsun/0625/3D-Mem/vis/target_obj_{pts[1]}.npy', np.array(obj["pcd"].points))
        

        return target_type, obj_pos, n_filtered_snapshots, target_index
    elif target_type == "object":
        target_index = int(target_index)
        if target_index not in list(scene.objects.keys()):
            logging.info(
                f"Predicted object index not in list: {target_index}, Choose a random frontier instead!"
            )
            return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
        # a = []
        # for idx in scene.objects.keys():
        #     if idx != target_index:
        #         a.append(scene.objects[idx]["pcd"].points)
        # np.save(f'/hsun/0625/3D-Mem/vis/other_obj_{pts[1]}.npy', np.concatenate(a, axis=0))
        # np.save(f'/hsun/0625/3D-Mem/vis/target_obj_{pts[1]}.npy', np.array(scene.objects[target_index]["pcd"].points))
        
        
        
        
        a = []
        for idx in scene.objects.keys():
            a.append(scene.objects[idx]["pcd"].points)
        target_point = Visibility_based_Viewpoint_Decision(
            np.array(scene.objects[target_index]["pcd"].points),
            np.concatenate(a, axis=0),
            pts,
            tsdf_planner,
            cfg.dicision_radius,
        )
        if target_point is None:
            target_point = select_navigation_corner(aabb = scene.objects[target_index]["bbox"], robot_position = pts)
            logging.info(f"Next choice: Object {target_point} (Closed Box Center)")
        else:
            logging.info(f"Next choice: Object {target_point} (Visible Center)")

        return target_type, target_point, n_filtered_snapshots, target_index
    else:  # target_type == "frontier"
        target_index = int(target_index)
        if target_index < 0 or target_index >= len(tsdf_planner.frontiers):
            logging.info(
                f"Predicted frontier target index out of range: {target_index}, Choose a random frontier instead!"
            )
            return random_frontier_choice(tsdf_planner, n_filtered_snapshots)
        target_point = tsdf_planner.frontiers[target_index].position
        logging.info(f"Next choice: Frontier at {target_point}")
        pred_target_frontier = tsdf_planner.frontiers[target_index]

        return target_type, pred_target_frontier, n_filtered_snapshots, target_index
    

def query_vlm_for_response_end(
    subtask_metadata: dict,
    rgb_egocentric_views: dict,
    cfg,
    verbose: bool = False,
):
    # prepare input for vlm
    step_dict = {}
    # prepare egocentric views
    step_dict["egocentric_views"] = rgb_egocentric_views
    step_dict["use_egocentric_views"] = True

    # prepare other metadata
    step_dict["question"] = subtask_metadata["question"]
    step_dict["task_type"] = subtask_metadata["task_type"]
    step_dict["class"] = subtask_metadata["class"]
    step_dict["image"] = subtask_metadata["image"]
    # query vlm
    (
        outputs,
        reason,
    ) = task_check(step_dict, verbose=verbose)

    logging.info(f"Response: [{outputs}]\nReason: [{reason}]")
    
    return outputs