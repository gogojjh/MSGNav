import os
import numpy as np
import logging
import random
import torch
import habitat_sim
import quaternion
import supervision as sv
import logging
from collections import Counter
from typing import List, Optional, Tuple, Dict, Union
import copy
from src.explore_utils_concept import get_edge_relationship
from habitat_sim.utils.common import (
    quat_to_coeffs,
    quat_from_angle_axis,
    quat_from_two_vectors,
)
from src.habitat import (
    make_semantic_cfg,
    get_quaternion,
    get_navigable_point_to,
)
from src.geom import get_cam_intr, IoU
from src.utils import rgba2rgb
# Local application/library specific imports
from src.conceptgraph.utils.ious import mask_subtract_contained
from src.conceptgraph.utils.general_utils import (
    ObjectClasses,
    measure_time,
    filter_detections,
)
import time
from src.conceptgraph.slam.slam_classes import MapEdge, MapObjectDict, DetectionDict, to_tensor
from src.conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    denoise_objects,
    merge_objects,
    detections_to_obj_pcd_and_bbox,
    processing_needed,
    resize_gobs,
    merge_obj2_into_obj1,
)
from src.conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
)
from src.conceptgraph.utils.model_utils import compute_clip_features_batched, clip_recognition, compute_clip_features_batched_check


class Scene:
    def __init__(
        self,
        scene_id,
        cfg,
        graph_cfg,
        detection_model,
        sam_predictor,
        clip_model,
        clip_preprocess,
        clip_tokenizer,
    ):
        self.cfg = cfg
        self.egde_time = 0
        self.edge_call = 0
        self.edge_token = 0
        # concept graph configuration
        self.cfg_cg = graph_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rooms = ['bedroom', 'living room', 'bathroom', 'kitchen room', 'laundry room', 'others']
        self.rooms_captions = '. '.join(self.rooms)+'.'
        # about the loading the scene
        if int(scene_id.split("-")[0]) < 800:
            split_path = os.path.join(cfg.scene_data_path, "train")
        else:
            split_path = os.path.join(cfg.scene_data_path, "val")
        scene_mesh_path = os.path.join(
            split_path, scene_id, scene_id.split("-")[1] + ".basis.glb"
        )
        navmesh_path = os.path.join(
            split_path, scene_id, scene_id.split("-")[1] + ".basis.navmesh"
        )
        semantic_texture_path = os.path.join(
            split_path, scene_id, scene_id.split("-")[1] + ".semantic.glb"
        )
        scene_semantic_annotation_path = os.path.join(
            split_path, scene_id, scene_id.split("-")[1] + ".semantic.txt"
        )
        assert os.path.exists(
            scene_mesh_path
        ), f"scene_mesh_path: {scene_mesh_path} does not exist"
        assert os.path.exists(
            navmesh_path
        ), f"navmesh_path: {navmesh_path} does not exist"
        assert os.path.exists(
            semantic_texture_path
        ), f"semantic_texture_path: {semantic_texture_path} does not exist"
        assert os.path.exists(
            scene_semantic_annotation_path
        ), f"scene_semantic_annotation_path: {scene_semantic_annotation_path} does not exist"

        sim_settings = {
            "scene": scene_mesh_path,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": cfg.img_width,
            "height": cfg.img_height,
            "hfov": cfg.hfov,
            "scene_dataset_config_file": cfg.scene_dataset_config_path,
            "camera_tilt": cfg.camera_tilt_deg * np.pi / 180,
        }
        sim_cfg = make_semantic_cfg(sim_settings)
        self.simulator = habitat_sim.Simulator(sim_cfg)
        self.pathfinder = self.simulator.pathfinder
        self.pathfinder.seed(cfg.seed)
        self.pathfinder.load_nav_mesh(navmesh_path)

        # load object classes
        # maintain a list of object classes
        self.obj_classes = ObjectClasses(
            classes_file_path=scene_semantic_annotation_path,
            bg_classes=self.cfg_cg["bg_classes"],
            skip_bg=self.cfg_cg["skip_bg"],
            class_set=self.cfg["class_set"],
        )

        logging.info(f"Load scene {scene_id} successfully")

        # set agent
        self.agent = self.simulator.initialize_agent(sim_settings["default_agent"])

        self.cam_intrinsic = get_cam_intr(cfg.img_width, cfg.img_height, cfg.hfov)

        # about scene graph
        self.objects: MapObjectDict[int, Dict] = (
            MapObjectDict()
        )  # object_id -> object item
        self.edges: MapEdge[(int,int), MapEdge] = (
            {}
        )  # edge_id -> edge item
        self.img_to_edge: Dict[str, List[(int, int)]] = {} 
        self.object_id_counter = 1

        self.all_observations: Dict[str, np.ndarray] = (
            {}
        )  # image_path -> image, stores all actual observations at each step, used for querying vlm
        self.all_obs_point: Dict[str, np.ndarray] = (
            {}
        )   # image_path -> point, stores the voxel point of the observation, used for planning
        self.all_depths = {}
        self.all_cam_poses = {}
        self.global_step_cnt = 0
        # setup detection and segmentation models
        self.detection_model = detection_model
        self.detection_model.set_classes(self.obj_classes.get_classes_arr())

        self.sam_predictor = sam_predictor

        self.clip_model = clip_model.to(self.device)
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer

    def __del__(self):
        try:
            self.simulator.close()
        except:
            pass

    def clear_up_detections(self):
        self.objects = MapObjectDict()
        self.object_id_counter = 1
        self.global_step_cnt = 0
        self.snapshots = {}
        self.all_observations = {}
        self.all_obs_point = {}
        self.all_depths = {}
        self.all_cam_poses = {}
        self.edges = ({})
        self.img_to_edge = {}
        


    def get_observation(self, pts, angle=None, rotation=None):
        assert (angle is None) ^ (
            rotation is None
        ), "Only one of angle and rotation should be specified"

        agent_state = habitat_sim.AgentState()
        agent_state.position = pts
        if angle is not None:
            agent_state.rotation = get_quaternion(angle, 0)
        else:
            agent_state.rotation = rotation
        self.agent.set_state(agent_state)

        obs = self.simulator.get_sensor_observations()

        # get camera extrinsic matrix
        sensor = self.agent.get_state().sensor_states["depth_sensor"]
        quaternion_0 = sensor.rotation
        translation_0 = sensor.position
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
        cam_pose[:3, 3] = translation_0

        obs["color_sensor"] = rgba2rgb(obs["color_sensor"])

        return obs, cam_pose

    def get_frontier_observation(self, pts, view_dir, camera_tilt=0.0):
        agent_state = habitat_sim.AgentState()

        # solve edge cases of viewing direction
        default_view_dir = np.asarray([0.0, 0.0, -1.0])
        if np.linalg.norm(view_dir) < 1e-3:
            view_dir = default_view_dir
        view_dir = view_dir / np.linalg.norm(view_dir)

        agent_state.position = pts
        # set agent observation direction
        if np.dot(view_dir, default_view_dir) / np.linalg.norm(view_dir) < -1 + 1e-3:
            # if the rotation is to rotate 180 degree, then the quaternion is not unique
            # we need to specify rotating along y-axis
            agent_state.rotation = quat_to_coeffs(
                quaternion.quaternion(0, 0, 1, 0)
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()
        else:
            agent_state.rotation = quat_to_coeffs(
                quat_from_two_vectors(default_view_dir, view_dir)
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()

        self.agent.set_state(agent_state)
        obs = self.simulator.get_sensor_observations()

        obs["color_sensor"] = rgba2rgb(obs["color_sensor"])

        return obs

    def get_frontier_observation_and_detect_target(
        self,
        pts,
        view_dir,
        detection_model,
        target_obj_id,
        target_obj_class,
        camera_tilt=0.0,
    ):
        obs = self.get_frontier_observation(pts, view_dir, camera_tilt)

        # detect target object
        rgb = obs["color_sensor"]
        semantic_obs = obs["semantic_sensor"]

        detection_model.set_classes([target_obj_class])
        results = detection_model.infer(
            rgb[..., :3], confidence=self.cfg.scene_graph.confidence
        )
        detections = sv.Detections.from_inference(results).with_nms(
            threshold=self.cfg.scene_graph.nms_threshold
        )

        target_detected = False
        if target_obj_id in np.unique(semantic_obs):
            for i in range(len(detections)):
                x_start, y_start, x_end, y_end = detections.xyxy[i].astype(int)
                bbox_mask = np.zeros(semantic_obs.shape, dtype=bool)
                bbox_mask[y_start:y_end, x_start:x_end] = True

                target_x_start, target_y_start = np.argwhere(
                    semantic_obs == target_obj_id
                ).min(axis=0)
                target_x_end, target_y_end = np.argwhere(
                    semantic_obs == target_obj_id
                ).max(axis=0)
                obj_mask = np.zeros(semantic_obs.shape, dtype=bool)
                obj_mask[target_x_start:target_x_end, target_y_start:target_y_end] = (
                    True
                )
                if IoU(bbox_mask, obj_mask) > self.cfg.scene_graph.iou_threshold:
                    target_detected = True
                    break

        return obs, target_detected

    def get_navigable_point_to(
        self,
        target_position,
        max_search=1000,
        min_dist=6.0,
        max_dist=999.0,
        prev_start_positions=None,
    ):
        self.pathfinder.seed(random.randint(0, 1000000))
        return get_navigable_point_to(
            target_position,
            self.pathfinder,
            max_search,
            min_dist,
            max_dist,
            prev_start_positions,
        )

    def update_scene_graph(
        self,
        image_rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics,
        cam_pos,
        pts,
        pts_voxel,
        img_path,
        frame_idx,
        semantic_obs=Optional[np.ndarray],
        gt_target_obj_ids=Optional[List[int]],
    ) -> Tuple[np.ndarray, List[int], Dict[int, int]]:
        # return annotated image; the detected object ids in current frame; the object id of the target object (if detected)
        assert not (
            (semantic_obs is None) ^ (gt_target_obj_ids is None)
        ), "semantic_obs and gt_target_obj_ids should be both None or both not None"
        self.intrinsics = intrinsics
        use_room_det = self.cfg.use_room_det
        if use_room_det:
            sim = compute_clip_features_batched_check(
                image_list=[image_rgb],
                clip_model=self.clip_model.to("cuda"),
                clip_tokenizer=self.clip_tokenizer,
                clip_preprocess=self.clip_preprocess,
                text_goal = np.array(self.rooms),
                image_goal = None,
                extra_text = False
            ).detach().cpu().numpy()
            room_label = self.rooms[np.argmax(sim)]
            room_conf = np.max(sim).item()
        else:
            room_label = 'unknown'
            room_conf = 0.0
        # set up object_classes first
        obj_classes = self.obj_classes

        # Detect objects
        results = self.detection_model.predict(image_rgb, conf=0.1, verbose=False)
        confidences = results[0].boxes.conf.cpu().numpy()
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        detection_class_labels = [
            f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
            for class_idx, class_id in enumerate(detection_class_ids)
        ]
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()
        # if there are detections,
        # Get Masks Using SAM or MobileSAM
        # UltraLytics SAM
        if xyxy_tensor.numel() != 0:
            sam_out = self.sam_predictor.predict(
                image_rgb, bboxes=xyxy_tensor, verbose=False
            )
            masks_tensor = sam_out[0].masks.data

            masks_np = masks_tensor.cpu().numpy()
        else:
            masks_np = np.empty((0, *image_rgb.shape[:2]), dtype=np.float64)

        # Create a detections object that we will save later
        curr_det = sv.Detections(
            xyxy=xyxy_np,
            confidence=confidences,
            class_id=detection_class_ids,
            mask=masks_np,
        )

        if len(curr_det) == 0:  # no detections, skip
            logging.debug("No detections in this frame")
            return image_rgb, [], {}

        # filter the detection by removing overlapping detections
        curr_det, labels = filter_detections(
            image=image_rgb,
            detections=curr_det,
            classes=obj_classes,
            given_labels=detection_class_labels,
            iou_threshold=self.cfg_cg.object_detection_iou_threshold,
            min_mask_size_ratio=self.cfg_cg.min_mask_size_ratio,
            confidence_threshold=self.cfg_cg.object_detection_confidence_threshold,
        )
        if curr_det is None:
            logging.debug("No detections left after filter_detections")
            return image_rgb, [], {}

        image_crops, image_feats, text_feats = compute_clip_features_batched(
            image_rgb,
            curr_det,
            self.clip_model,
            self.clip_preprocess,
            self.clip_tokenizer,
            obj_classes.get_classes_arr(),
            self.device,
        )

        raw_gobs = {
            # add new uuid for each detection
            "xyxy": curr_det.xyxy,
            "confidence": curr_det.confidence,
            "class_id": curr_det.class_id,
            "mask": curr_det.mask,
            "classes": obj_classes.get_classes_arr(),
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
            "detection_class_labels": detection_class_labels,
            "room_label": room_label,
            "room_conf": room_conf,
        }

        # resize the observation if needed
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        # filter the observations
        filtered_gobs = filter_gobs(
            resized_gobs,
            image_rgb,
            skip_bg=self.cfg_cg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=self.cfg_cg.mask_area_threshold,
            max_bbox_area_ratio=self.cfg_cg.max_bbox_area_ratio,
            mask_conf_threshold=self.cfg_cg.mask_conf_threshold,
        )

        gobs = filtered_gobs

        if len(gobs["mask"]) == 0:  # no detections in this frame
            logging.debug("No detections left after filter_gobs")
            return image_rgb, [], {}

        # this helps make sure things like pillows on couches are separate objects
        gobs["mask"] = mask_subtract_contained(gobs["xyxy"], gobs["mask"])

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth,
            masks=gobs["mask"],
            cam_K=intrinsics[:3, :3],  # Camera intrinsics
            image_rgb=image_rgb,
            trans_pose=cam_pos,
            min_points_threshold=self.cfg_cg.min_points_threshold,
            spatial_sim_type=self.cfg_cg.spatial_sim_type,
            obj_pcd_max_points=self.cfg_cg.obj_pcd_max_points,
            device=self.device,
        )

        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                    dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                    dbscan_eps=self.cfg_cg["dbscan_eps"],
                    dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                    pcd=obj["pcd"],
                )

        # for obj in self.objects:
        #     points.append(np.asarray(obj["pcd"].points))
        #     obj["bbox"] = get_bounding_box(
        #         spatial_sim_type=self.cfg_cg["spatial_sim_type"],
        #         pcd=obj["pcd"],
        #     )
        #     boxes.append(obj["bbox"].get_box_points())
        # points = np.concatenate(points, axis = 0)
        # boxes = np.array(boxes)
        # print(points.shape, boxes.shape)
        # np.save('/data0/hsun/7-15/3D-Mem/vis/points.npy',points)
        # np.save('/data0/hsun/7-15/3D-Mem/vis/boxes.npy',boxes)
        # if the list is all None, then skip
        if all([obj is None for obj in obj_pcds_and_bboxes]):
            logging.debug("All objects are None in obj_pcds_and_bboxes")
            return image_rgb, [], {}

        # add pcds and bboxes to gobs
        gobs["bbox"] = [
            obj["bbox"] if obj is not None else None for obj in obj_pcds_and_bboxes
        ]
        gobs["pcd"] = [
            obj["pcd"] if obj is not None else None for obj in obj_pcds_and_bboxes
        ]

        # filter out objects that are far away
        gobs = self.filter_gobs_with_distance(pts, gobs)

        detection_list = self.make_detection_list_from_pcd_and_gobs(
            gobs, img_path, obj_classes
        )

        if len(detection_list) == 0:  # no detections, skip
            logging.debug(
                "No detections left after make_detection_list_from_pcd_and_gobs"
            )
            return image_rgb, [], {}

        # compare the detections with the target object mask to see whether the target object is detected
        target_obj_id_mapping = {}
        if semantic_obs is not None:
            for target_gt_id in gt_target_obj_ids:#####target groundtruth
                target_obj_mask = semantic_obs == target_gt_id
                if (
                    np.sum(target_obj_mask)
                    / (target_obj_mask.shape[0] * target_obj_mask.shape[1])
                    > 0.0001
                ):
                    # loop through the detected objects to find the highest IoU with the target object
                    max_iou = -1
                    max_iou_obj_id = None
                    for idx, obj_id in enumerate(detection_list.keys()):
                        detected_mask = gobs["mask"][idx]
                        iou_score = IoU(detected_mask, target_obj_mask)
                        if iou_score > max_iou:
                            max_iou = iou_score
                            max_iou_obj_id = obj_id
                    if max_iou > self.cfg.scene_graph.target_obj_iou_threshold:#####some detected obj match!
                        target_obj_id_mapping[target_gt_id] = max_iou_obj_id
                        logging.info(
                            f"Target object {target_gt_id} detected with IoU {max_iou} in {img_path}!!!"
                        )


        det_visual_prompt = sv.Detections(
            xyxy=gobs["xyxy"],
            class_id=gobs["class_id"],
        )

        # if no objects yet in the map,
        # just add all the objects from the current frame
        # then continue, no need to match or merge
        if len(self.objects) == 0:
            logging.debug(
                f"No objects in the map yet, adding all detections of length {len(detection_list)}"
            )
            self.objects.update(detection_list)

            det_visual_prompt.data["obj_id"] = list(detection_list.keys())


            annotated_image = image_rgb
            added_obj_ids = list(detection_list.keys())
            self.update_scene_graph_edges(
                frame_obj_ids=added_obj_ids,
                img_path=img_path,
            )
            #update_scene_graph_edges_concept
        else:
            ### compute similarities and then merge
            spatial_sim = compute_spatial_similarities(
                spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                detection_list=detection_list,
                objects=self.objects,
                downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
            )

            visual_sim = compute_visual_similarities(detection_list, self.objects)

            agg_sim = aggregate_similarities(
                match_method=self.cfg_cg["match_method"],
                phys_bias=self.cfg_cg["phys_bias"],
                spatial_sim=spatial_sim,
                visual_sim=visual_sim,
            )

            # Perform matching of detections to existing objects
            match_indices = match_detections_to_objects(
                agg_sim=agg_sim,
                detection_threshold=self.cfg_cg[
                    "sim_threshold"
                ],  # Use the sim_threshold from the configuration
                existing_obj_ids=list(self.objects.keys()),
                detected_obj_ids=list(detection_list.keys()),
            )

            # Now merge the detected objects into the existing objects based on the match indices
            visualize_captions, target_obj_id_mapping, added_obj_ids, all_obj_ids = (
                self.merge_obj_matches(
                    detection_list=detection_list,
                    match_indices=match_indices,
                    obj_classes=obj_classes,
                    target_obj_id_mapping=target_obj_id_mapping,
                )
            )
            self.update_scene_graph_edges(######update scene graph edges
                frame_obj_ids=list(set(all_obj_ids)),
                img_path=img_path,
            )
            det_visual_prompt.data["obj_id"] = all_obj_ids


            # create a Detection object for visualization
            det_visualize = sv.Detections(
                xyxy=gobs["xyxy"],
                confidence=gobs["confidence"],
                class_id=gobs["class_id"],
            )
            det_visualize.data["class_name"] = visualize_captions
            annotated_image = image_rgb.copy()
            BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
            LABEL_ANNOTATOR = sv.LabelAnnotator(
                text_thickness=1, text_scale=0.25, text_color=sv.Color.BLACK
            )
            annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(
                annotated_image, det_visualize
            )
            annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, det_visualize)
        
        return annotated_image, added_obj_ids, target_obj_id_mapping
    
    def update_scene_graph_edges(self, frame_obj_ids, img_path: str):
        """
        Update the edges in the scene graph based on the detection list.
        This function will create new edges for each object in the detection list
        and update the existing edges if necessary.
        """
        self.img_to_edge[img_path] = []
        for a_obj_id in frame_obj_ids:
            for b_obj_id in frame_obj_ids:
                if a_obj_id > b_obj_id:
                    # only create edges for pairs (obj_id, added_obj_id) where obj_id <= added_obj_id
                    continue
                obj1_center = self.objects[a_obj_id]["bbox"].center[[0, 2]]
                obj2_center = self.objects[b_obj_id]["bbox"].center[[0, 2]]
                if np.linalg.norm(obj1_center - obj2_center) > self.cfg_cg.edge_dist_threshold:
                    # if the distance between the two objects is larger than the threshold, skip
                    continue

                if (a_obj_id, b_obj_id) in self.edges.keys():
                    #Shared edge, only require one insertion
                    self.edges[(a_obj_id, b_obj_id)].rel_img.append(img_path) 
                    self.edges[(a_obj_id, b_obj_id)].num_detections += 1
                    self.img_to_edge[img_path].append((a_obj_id, b_obj_id))
                    if a_obj_id != b_obj_id:
                        self.img_to_edge[img_path].append((b_obj_id, a_obj_id))
                else:   
                    # Create a new edge for this object
                    edge = MapEdge(
                        obj1_idx=a_obj_id,
                        obj2_idx=b_obj_id,
                        rel_img=img_path,
                        num_detections=1,
                    )
                    self.edges[(a_obj_id, b_obj_id)] = edge
                    self.img_to_edge[img_path].append((a_obj_id, b_obj_id))
                    if a_obj_id != b_obj_id:#####b-a is also an edge, add
                        self.edges[(b_obj_id, a_obj_id)] = edge
                        self.img_to_edge[img_path].append((b_obj_id, a_obj_id))
        
    def update_scene_graph_edges_concept(self, frame_obj_ids, img_path: str):
        """
        Update the edges in the scene graph based on the detection list.
        This function will create new edges for each object in the detection list
        and update the existing edges if necessary.
        """
        begin_time = time.time()
        self.img_to_edge[img_path] = []
        object_infos = ''
        edge_infos = ''
        edge_list = []
        for a_obj_id in frame_obj_ids:
            object_infos += f"Object {a_obj_id}: 1.bbox_extent: ({self.objects[a_obj_id]['bbox'].extent[0]:.2f}, {self.objects[a_obj_id]['bbox'].extent[2]:.2f}, {self.objects[a_obj_id]['bbox'].extent[1]:.2f}), 2.bbox_center: ({self.objects[a_obj_id]['bbox'].center[0]:.2f}, {self.objects[a_obj_id]['bbox'].center[2]:.2f}, {self.objects[a_obj_id]['bbox'].center[1]:.2f}), 3.object_tag: {self.objects[a_obj_id]['class_name']}\n"
            for b_obj_id in frame_obj_ids:
                if a_obj_id > b_obj_id:
                    # only create edges for pairs (obj_id, added_obj_id) where obj_id <= added_obj_id
                    continue
                obj1 = self.objects[a_obj_id]
                obj2 = self.objects[b_obj_id]
                obj1_center = self.objects[a_obj_id]["bbox"].center[[0, 2]]
                obj2_center = self.objects[b_obj_id]["bbox"].center[[0, 2]]
                if np.linalg.norm(obj1_center - obj2_center) > self.cfg_cg.edge_dist_threshold:
                    # if the distance between the two objects is larger than the threshold, skip
                    continue

                if (a_obj_id, b_obj_id) in self.edges.keys():
                    #Shared edge, only require one insertion
                    self.edges[(a_obj_id, b_obj_id)].rel_img.append(img_path) 
                    self.edges[(a_obj_id, b_obj_id)].num_detections += 1
                    self.img_to_edge[img_path].append((a_obj_id, b_obj_id))
                    if a_obj_id != b_obj_id:
                        self.img_to_edge[img_path].append((b_obj_id, a_obj_id))
                        if self.edges[(a_obj_id, b_obj_id)].caption is None and self.edges[(b_obj_id, a_obj_id)].caption is None:
                            edge_list.append((a_obj_id, b_obj_id))
                            edge_infos += f"{a_obj_id} {b_obj_id}\n"
                else:
                    edge = MapEdge(
                        obj1_idx=a_obj_id,
                        obj2_idx=b_obj_id,
                        rel_img=img_path,
                        num_detections=1
                    )
                    self.edges[(a_obj_id, b_obj_id)] = edge
                    self.img_to_edge[img_path].append((a_obj_id, b_obj_id))
                    if a_obj_id != b_obj_id:
                        edge_list.append((a_obj_id, b_obj_id))
                        edge_infos += f"{a_obj_id} {b_obj_id}\n"
                        self.edges[(b_obj_id, a_obj_id)] = edge
                        self.img_to_edge[img_path].append((b_obj_id, a_obj_id))
                    else:
                        self.edges[(a_obj_id, b_obj_id)].caption = f"The same object."

        DEFAULT_PROMPT = """The list of object-pairs describing objects. You need to produce a object_relation for each object-pair.
Each of the object will have following fields:
1. bbox_extent: the 3D bounding box extents of the object
2. bbox_center: the 3D bounding box center of the object
3. object_tag: an extremely brief description of the object

The "object_relation" field must be one of the following (reply index):
1. if object a is an object commonly placed on top of object b
2. if object b is an object commonly placed on top of object a
3. if object a is an object commonly placed inside object b
4. if object b is an object commonly placed inside object a
5. If object a is an object commonly placed just beside object b

Here is a example, you must follow the format strictly:
Objects: 
Object 1: 1.bbox_extent: (0.50, 0.50, 0.50), 2.bbox_center: (1.00, 1.00, 1.00), 3.object_tag: cup
Object 2: 1.bbox_extent: (1.00, 1.00, 1.00), 2.bbox_center: (1.00, 0.50, 1.00), 3.object_tag: table
Object 3: 1.bbox_extent: (2.00, 2.00, 2.00), 2.bbox_center: (5.00, 1.00, 5.00), 3.object_tag: bed

Object-pairs: 
1 2
1 3

Answer:
1 
5

The followings the concrete content of the task.
Objects: 
"""
        DEFAULT_PROMPT += object_infos
        DEFAULT_PROMPT += '\nObject-pairs: \n'
        DEFAULT_PROMPT += edge_infos
        if len(edge_infos) == 0:
            DEFAULT_PROMPT += 'No Object-pair'
        DEFAULT_PROMPT += "\n\nAnswer (only reply matched relationship index of object_relation for each object-pair):\n"
        DEFAULT_PROMPT += "Do not include any additional information other than the index in your response"
        response = get_edge_relationship(DEFAULT_PROMPT)
        if '\n' in response:
            relations = response.split("\n")
        else:
            relations = response.split(" ")
        if len(relations) == len(edge_list):
            for idx, node in enumerate(edge_list):
                relation = relations[idx].strip()
                if relation.isdigit() and int(relation) == 1:
                    caption = f"{node[0]} is commonly placed on top of {node[1]}"
                elif relation.isdigit() and int(relation) == 2:
                    caption = f"{node[1]} is commonly placed on top of {node[0]}"
                elif relation.isdigit() and int(relation) == 3:
                    caption = f"{node[0]} is commonly placed inside {node[1]}"
                elif relation.isdigit() and int(relation) == 4:
                    caption = f"{node[1]} is commonly placed inside {node[0]}"
                else:
                    caption = f"{node[0]} is commonly placed just beside {node[1]}"
                self.edges[node].caption = caption
        self.edge_token += self.tokenizercounter.count_tokens(DEFAULT_PROMPT)
        self.edge_call += 1
        self.egde_time += time.time() - begin_time
        if self.edge_call%20 == 5:
            logging.info(f"Average Token Cost : {self.edge_token / self.edge_call}")
            logging.info(f"Average Time Cost : {self.egde_time / self.edge_call}")
            logging.info(f"Total Call : {self.edge_call}")


    def del_unused_scene_graph_edges(self):
        # clear up the edges that are not in the current objects
        edges_to_remove = []
        for edge_key in self.edges.keys():
            obj1_idx, obj2_idx = edge_key
            if obj1_idx not in self.objects.keys() or obj2_idx not in self.objects.keys():
                edges_to_remove.append(edge_key)
        for edge_key in edges_to_remove:#####remove scene graph edges
            if edge_key in self.edges.keys():
                del self.edges[edge_key]
            if (edge_key[1], edge_key[0]) in self.edges.keys():
                del self.edges[(edge_key[1], edge_key[0])]


    def filter_gobs_with_distance(self, pts, gobs):
        idx_to_keep = []
        for idx in range(len(gobs["bbox"])):
            if gobs["bbox"][idx] is None:  # point cloud was discarded
                continue

            # get the distance between the object and the current observation point
            if (
                np.linalg.norm(gobs["bbox"][idx].center[[0, 2]] - pts[[0, 2]])
                > self.cfg.scene_graph.obj_include_dist
            ):
                logging.debug(
                    f"Object {gobs['detection_class_labels'][idx]} is too far away, skipping"
                )
                continue
            idx_to_keep.append(idx)

        for attribute in gobs.keys():
            if isinstance(gobs[attribute], str) or isinstance(gobs[attribute], float) or attribute == "classes":  # Captions
                continue
            if attribute in ["labels", "edges", "text_feats", "captions"]:
                # Note: this statement was used to also exempt 'detection_class_labels' but that causes a bug. It causes the edges to be misalgined with the objects.
                continue
            elif isinstance(gobs[attribute], list):
                gobs[attribute] = [gobs[attribute][i] for i in idx_to_keep]
            elif isinstance(gobs[attribute], np.ndarray):
                gobs[attribute] = gobs[attribute][idx_to_keep]
            else:
                raise NotImplementedError(f"Unhandled type {type(gobs[attribute])}")

        return gobs

    def merge_obj_matches(
        self,
        detection_list: DetectionDict,
        match_indices: List[Tuple[int, Optional[int]]],
        obj_classes: ObjectClasses,
        target_obj_id_mapping: Dict[int, int],
    ) -> Tuple[List[str], Dict[int, int], List[int], List[int]]:
        visualize_captions = []
        all_obj_ids = []
        added_obj_ids = []
        for idx, (detected_obj_id, existing_obj_match_id) in enumerate(match_indices):
            if existing_obj_match_id is None:
                self.objects[detected_obj_id] = detection_list[detected_obj_id]
                visualize_captions.append(
                    f"{detected_obj_id} {self.objects[detected_obj_id]['class_name']} {self.objects[detected_obj_id]['conf']:.3f} N"
                )
                all_obj_ids.append(detected_obj_id)
                added_obj_ids.append(detected_obj_id)
            else:
                # merge detected object into existing object
                detected_obj = detection_list[detected_obj_id]
                matched_obj = self.objects[existing_obj_match_id]

                merged_obj = merge_obj2_into_obj1(
                    obj1=matched_obj,
                    obj2=detected_obj,
                    downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                    dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                    dbscan_eps=self.cfg_cg["dbscan_eps"],
                    dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                    spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                    device=self.device,
                    run_dbscan=False,
                )
                # fix the class name by adopting the most popular class name
                class_id_counter = Counter(merged_obj["class_id"])
                most_common_class_id = class_id_counter.most_common(1)[0][0]
                most_common_class_name = obj_classes.get_classes_arr()[
                    most_common_class_id
                ]
                merged_obj["class_name"] = most_common_class_name

                self.objects[existing_obj_match_id] = merged_obj
                visualize_captions.append(
                    f"{existing_obj_match_id} {self.objects[existing_obj_match_id]['class_name']} {detected_obj['conf']:.3f} {merged_obj['num_detections']}"
                )
                all_obj_ids.append(existing_obj_match_id)

                # update the mapping of target object id
                for gt_id, mapped_id in target_obj_id_mapping.items():
                    if mapped_id == detected_obj_id:
                        target_obj_id_mapping[gt_id] = existing_obj_match_id

        return visualize_captions, target_obj_id_mapping, added_obj_ids, all_obj_ids

    def make_detection_list_from_pcd_and_gobs(
        self, gobs, image_path, obj_classes
    ) -> DetectionDict:
        detection_list = DetectionDict()
        for mask_idx in range(len(gobs["mask"])):
            if gobs["pcd"][mask_idx] is None:  # point cloud was discarded
                continue

            curr_class_name = gobs["classes"][gobs["class_id"][mask_idx]]
            curr_class_idx = obj_classes.get_classes_arr().index(curr_class_name)

            detected_object = {
                "id": self.object_id_counter,  # unique id for this object
                "class_name": curr_class_name,  # global class id for this detection
                "class_id": [curr_class_idx],  # global class id for this detection
                "num_detections": 1,  # number of detections in this object
                "conf": gobs["confidence"][mask_idx],
                # These are for the entire 3D object
                "pcd": gobs["pcd"][mask_idx],
                "bbox": gobs["bbox"][mask_idx],
                "mask_num": gobs["mask"][mask_idx].sum(),
                "clip_ft": to_tensor(gobs["image_feats"][mask_idx]),
                # the image name it belongs to
                "image": None,
                "room_label" : gobs["room_label"],
                "room_conf" : gobs["room_conf"],
                "image_path" : image_path,  # the image path it belongs to
                "image_path_list" : [image_path],  # the image path it belongs to
            }

            detection_list[self.object_id_counter] = detected_object
            self.object_id_counter += 1

        return detection_list

    def periodic_cleanup_objects(self, frame_idx, pts, goal_obj_ids_mapping=None):
        ### Perform post-processing periodically if told so

        # Denoising
        if processing_needed(
            self.cfg_cg["denoise_interval"],
            self.cfg_cg["run_denoise_final_frame"],
            frame_idx,
            is_final_frame=False,
        ):
            self.objects = measure_time(denoise_objects)(
                downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                dbscan_eps=self.cfg_cg["dbscan_eps"],
                dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                device=self.device,
                objects=self.objects,
            )

        # Filtering
        if processing_needed(
            self.cfg_cg["filter_interval"],
            self.cfg_cg["run_filter_final_frame"],
            frame_idx,
            is_final_frame=False,
        ):
            self.objects = filter_objects(
                obj_min_points=self.cfg_cg["obj_min_points"],
                obj_min_detections=self.cfg_cg["obj_min_detections"],
                min_distance=self.cfg.scene_graph.obj_include_dist,
                objects=self.objects,
                pts=pts,
            )

        # Merging
        if processing_needed(
            self.cfg_cg["merge_interval"],
            self.cfg_cg["run_merge_final_frame"],
            frame_idx,
            is_final_frame=False,
        ):
            self.objects = measure_time(merge_objects)(
                merge_overlap_thresh=self.cfg_cg["merge_overlap_thresh"],
                merge_visual_sim_thresh=self.cfg_cg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=self.cfg_cg["merge_text_sim_thresh"],
                objects=self.objects,
                downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                dbscan_eps=self.cfg_cg["dbscan_eps"],
                dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                device=self.device,
                goal_obj_ids_mapping=goal_obj_ids_mapping,
            )


        # update the goal object ids mapping to remove the objects that have been removed
        if goal_obj_ids_mapping is not None:
            for goal_obj_id, mapped_obj_ids in goal_obj_ids_mapping.items():
                goal_obj_ids_mapping[goal_obj_id] = [
                    obj_id for obj_id in mapped_obj_ids if obj_id in self.objects.keys()
                ]

    def sanity_check(self, cfg):
        
        pass