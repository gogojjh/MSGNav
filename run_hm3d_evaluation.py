import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"

import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import math
import time
import json
import logging
import matplotlib.pyplot as plt

import open_clip
from ultralytics import SAM, YOLOWorld

from src.habitat import pose_habitat_to_tsdf
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_planner import TSDFPlanner, Frontier
from src.multimodal_3d_scene_graph import Scene
from src.utils import resize_image, calc_agent_subtask_distance, get_pts_angle_goatbench
from src.query_vlm import query_vlm_for_response, query_vlm_for_response_end
from src.logger_hm3d import Logger
import time

class Timer:
    def __init__(self):
        self.start_time = {}
        self.total_time = {}
        self.count = {}

    def start(self, module_name):
        self.start_time[module_name] = time.time()
        if module_name not in self.total_time:
            self.total_time[module_name] = 0.0
            self.count[module_name] = 0

    def stop(self, module_name):
        if module_name not in self.start_time:
            raise ValueError(f"Module '{module_name}' was not started.")
        
        end_time = time.time()
        elapsed_time = end_time - self.start_time[module_name]
        
        self.total_time[module_name] += elapsed_time
        self.count[module_name] += 1

    def get_average_time(self, module_name):
        if module_name not in self.total_time:
            raise ValueError(f"Module '{module_name}' does not exist.")
        
        if self.count[module_name] == 0:
            return 0.0
        return self.total_time[module_name] / self.count[module_name]

    def reset(self, module_name=None):
        if module_name is None:
            self.start_time = {}
            self.total_time = {}
            self.count = {}
        elif module_name in self.total_time:
            del self.start_time[module_name]
            del self.total_time[module_name]
            del self.count[module_name]
        else:
            raise ValueError(f"Module '{module_name}' does not exist.")

    def print_summary(self, logger):
        logging.info("Time counter:")
        idx = 0
        sum = 0
        for module in self.total_time:
            avg_time = self.get_average_time(module)
            sum += avg_time

        logging.info(f" total average {sum:.6f}s / per step")
        for module in self.total_time:
            avg_time = self.get_average_time(module)
            idx += 1
            logging.info(f" {idx} {module}: average {avg_time:.6f} s")


def main(cfg, start_ratio=0.0, end_ratio=1.0, specific = None):
    # load the default concept graph config
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Load dataset
    scene_data_list = ['4ok3usBNeis.json', '5cdEh9F2hJL.json', '6s7QHgap2fW.json', '7MXmsvcQjpJ.json', 'BAbdmeyTvMZ.json', 'CrMo8WxCyVb.json', 'DYehNKdT76V.json', 'Dd4bFSTQ8gi.json', 'GLAQ4DNUx5U.json', 'HY1NcmCgn3n.json', 'LT9Jq6dN3Ea.json', 'MHPLjHsuG27.json', 'Nfvxx8J5NCo.json', 'QaLdnwvtxbs.json', 'TEEsavR23oF.json', 'VBzV5z6i1WS.json', 'XB4GS9ShBRE.json', 'a8BtkwhxdRV.json', 'bCPU9suPUw9.json', 'bxsVRursffK.json', 'cvZr5TUy5C5.json', 'eF36g7L6Z9M.json', 'h1zeeAwLh9Z.json', 'k1cupFYWXJ6.json', 'mL8ThkuaVTM.json', 'mv2HUxq3B53.json', 'p53SfW6mjZe.json', 'q3zU7Yy5E5s.json', 'q5QZSEeHe5g.json', 'qyAac8rV8Zk.json', 'svBbv1Pavdk.json', 'wcojb4TFT35.json', 'y9hTuugGdiq.json', 'yr17PDCnDDW.json', 'ziup5kvtCCR.json', 'zt1RVoi7PcG.json']
    scene_data_list = os.listdir(cfg.test_data_dir)
    num_scene = len(scene_data_list)
    random.shuffle(scene_data_list)
    num_episode = 0
    
    # split the test data by scene
    if specific is not None:
        scene_data_list = scene_data_list[specific:specific + 1]
        logging.info(f"Specific scene {specific} selected for evaluation.")
    else:
        scene_data_list = scene_data_list[
            int(start_ratio * num_scene) : int(end_ratio * num_scene)
        ]
    for scene_data_file in scene_data_list:
        with open(os.path.join(cfg.test_data_dir, scene_data_file), "r") as f:
            num_episode += len(json.load(f)["episodes"])
    logging.info(f"Total number of episodes: {num_episode}")

    all_scene_ids = os.listdir(cfg.scene_data_path + "/train") + os.listdir(
        cfg.scene_data_path + "/val"
    )

    # load detection and segmentation models
    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

    sam_predictor = SAM(cfg.sam_model_name)  # UltraLytics SAM
    logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

    # clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    #     "ViT-B-32", "laion2b_s34b_b79k"  # "ViT-H-14", "laion2b_s32b_b79k"
    # )
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14-quickgelu", "dfn5b"  # "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")
    logging.info(f"Load CLIP model successful!")

    # Initialize the logger
    logger = Logger(
        cfg.output_dir, start_ratio, end_ratio, voxel_size=cfg.tsdf_grid_size, specific=specific
    )
    timer = Timer()
    scene_idx = 0
    global_episode = 0
    for scene_data_file in scene_data_list:
        scene_idx += 1
        # load goatbench data
        scene_name = scene_data_file.split(".")[0]
        scene_id = [scene_id for scene_id in all_scene_ids if scene_name in scene_id][0]
        scene_data = json.load(
            open(os.path.join(cfg.test_data_dir, scene_data_file), "r")
        )
        total_episodes = len(scene_data["episodes"])

        navigation_goals = scene_data[
            "goals_by_category"
        ]  # obj_id to obj_data, apply for all episodes in this scene
        logging.info(f"Loading scene {scene_id}")
        navigation_goals = dict([(k.split('glb_')[-1], v) for k, v in navigation_goals.items()])
        try:
            del scene
        except:
            pass
        scene = Scene(
            scene_id,
            cfg,
            cfg_cg,
            detection_model,
            sam_predictor,
            clip_model,
            clip_preprocess,
            clip_tokenizer,
        )
        for episode_idx, cur_episode in enumerate(scene_data["episodes"]):
            global_episode += 1


            finished_subtask_ids = list(logger.success_by_distance.keys())
            episode_id = str(episode_idx)
            curr_key = str(scene_id)+'_'+str(episode_id)


            if curr_key in finished_subtask_ids:
                logging.info(f"Scene {scene_id} Episode {episode_idx} already done!")
                continue
            
            scene.clear_up_detections()

            logging.info(f"Episode {episode_idx + 1}/{total_episodes}")
            goals = navigation_goals[cur_episode['object_category']]
            goal_types = ['object']

            # check whether this episode has been processed
            

            pts, angle = get_pts_angle_goatbench(
                cur_episode["start_position"], cur_episode["start_rotation"]
            )

            # load scene
            
            
            # initialize the TSDF
            floor_height = pts[1]
            tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
            num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
            num_step = max(num_step, 50)
            tsdf_planner = TSDFPlanner(
                vol_bnds=tsdf_bnds,
                voxel_size=cfg.tsdf_grid_size,
                floor_height=floor_height,
                floor_height_offset=0,
                pts_init=pts,
                init_clearance=cfg.init_clearance * 2,
                save_visualization=cfg.save_visualization,
            )

            episode_dir, eps_frontier_dir = logger.init_episode(
                episode_id=f"{scene_id}_ep_{episode_id}"
            )

            logging.info(f"\nScene {scene_id} initialization successful!")

            subtask_metadata = logger.init_subtask(
                object_category=cur_episode['object_category'],
                episode_id = episode_id,
                goals=goals,
                pts=pts,
                scene=scene,
                tsdf_planner=tsdf_planner,
            )

            # mapping from the obj id in habitat to the id assigned by concept graph
            # this mapping/alignment is done by heuristic matching between object masks
            goal_obj_ids_mapping = {
                obj_id: [] for obj_id in subtask_metadata["goal_obj_ids"]
            }

            # run steps
            task_success = False
            task_check_obs = []
            cnt_step = -1
            his_decision = {}
            subtask_metadata['CLR'] = his_decision
            # reset tsdf planner
            tsdf_planner.max_point = None
            tsdf_planner.target_point = None
            max_point_choice = None

            

            while cnt_step < num_step - 1:
                decision = {}
                cnt_step += 1
                his_decision['cnt_step'] = cnt_step
                his_decision['max_step'] = num_step
                logging.info(
                    f"\n== Scene: {scene_idx}, Episode: {global_episode}, Step: {cnt_step}=="
                )
                timer.start("Observe the surroundings")
                # (1) Observe the surroundings, update the scene graph and occupancy map
                # Determine the viewing angles for the current step
                if cnt_step == 0:
                    angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180
                    total_views = 1 + cfg.extra_view_phase_2
                else:
                    angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180
                    total_views = 1 + cfg.extra_view_phase_1
                all_angles = [
                    angle + angle_increment * (i - total_views // 2)
                    for i in range(total_views)
                ]
                # Let the main viewing angle be the last one to avoid potential overwriting problems
                main_angle = all_angles.pop(total_views // 2)
                all_angles.append(main_angle)

                rgb_egocentric_views = []
                all_added_obj_ids = (
                    []
                )  # Record all the objects that are newly added in this step
                for view_idx, ang in enumerate(all_angles):
                    # For each view
                    obs, cam_pose = scene.get_observation(pts, angle=ang)
                    rgb = obs["color_sensor"]
                    depth = obs["depth_sensor"]
                    semantic_obs = obs["semantic_sensor"]

                    # collect all view features
                    obs_file_name = f"{cnt_step}-view_{view_idx}.png"
                    with torch.no_grad():
                        # Concept graph pipeline update
                        annotated_rgb, added_obj_ids, target_obj_id_mapping = (
                            scene.update_scene_graph(
                                image_rgb=rgb[..., :3],
                                depth=depth,
                                intrinsics=cam_intr,
                                cam_pos=cam_pose,
                                pts=pts,
                                pts_voxel=tsdf_planner.habitat2voxel(pts),
                                img_path=obs_file_name,
                                frame_idx=cnt_step * total_views + view_idx,
                                semantic_obs=semantic_obs,
                                gt_target_obj_ids=subtask_metadata["goal_obj_ids"],
                            )
                        )
                        scene.all_observations[obs_file_name] = rgb
                        scene.all_depths[obs_file_name] = depth
                        scene.all_cam_poses[obs_file_name] = cam_pose
                        scene.all_obs_point[obs_file_name] = tsdf_planner.habitat2voxel(pts)
                        rgb_egocentric_views.append(
                            resize_image(rgb, cfg.prompt_h, cfg.prompt_w)
                        )
                        for (
                            gt_goal_id,
                            det_goal_id,
                        ) in target_obj_id_mapping.items():
                            goal_obj_ids_mapping[gt_goal_id].append(det_goal_id)
                        all_added_obj_ids += added_obj_ids

                    scene.periodic_cleanup_objects(
                        frame_idx=cnt_step * total_views + view_idx,
                        pts=pts,
                        goal_obj_ids_mapping=goal_obj_ids_mapping,
                    )

                    tsdf_planner.integrate(
                        color_im=rgb,
                        depth_im=depth,
                        cam_intr=cam_intr,
                        cam_pose=pose_habitat_to_tsdf(cam_pose),
                        obs_weight=1.0,
                        margin_h=int(cfg.margin_h_ratio * img_height),
                        margin_w=int(cfg.margin_w_ratio * img_width),
                        explored_depth=cfg.explored_depth,
                    )
                logging.info(f"Goal object mapping: {goal_obj_ids_mapping}")
                timer.stop("Observe the surroundings")
                
                timer.start("Update Memory")
                all_added_obj_ids = [
                    obj_id
                    for obj_id in all_added_obj_ids
                    if obj_id in scene.objects
                ]
                for obj_id, obj in scene.objects.items():
                    if (
                        np.linalg.norm(obj["bbox"].center[[0, 2]] - pts[[0, 2]])
                        < cfg.scene_graph.obj_include_dist + 0.5
                    ):
                        all_added_obj_ids.append(obj_id)
                    
                scene.del_unused_scene_graph_edges()
                timer.stop("Update Memory")

                timer.start("Update Frontier")
                update_success = tsdf_planner.update_frontier_map(
                    pts=pts,
                    cfg=cfg.planner,
                    scene=scene,
                    cnt_step=cnt_step,
                    save_frontier_image=cfg.save_visualization,
                    eps_frontier_dir=eps_frontier_dir,
                    prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
                )
                if not update_success:
                    logging.info("Warning! Update frontier map failed!")
                timer.stop("Update Frontier")


                timer.start("Querying the VLM")
                if cfg.choose_every_step:
                    # if we choose to query vlm every step, we clear the target point every step
                    if (
                        tsdf_planner.max_point is not None
                        and type(tsdf_planner.max_point) == Frontier
                    ):
                        # reset target point to allow the model to choose again
                        tsdf_planner.max_point = None
                        tsdf_planner.target_point = None

                # use the most common id in the mapped ids as the detected target object id
                target_obj_ids_estimate = []
                for obj_id, det_ids in goal_obj_ids_mapping.items():
                    if len(det_ids) == 0:
                        continue
                    target_obj_ids_estimate.append(
                        max(set(det_ids), key=det_ids.count)
                    )

                if (
                    tsdf_planner.max_point is None
                    and tsdf_planner.target_point is None
                ):
                    # query the VLM for the next navigation point, and the reason for the choice
                    
                    vlm_response = query_vlm_for_response(
                        subtask_metadata=subtask_metadata,
                        scene=scene,
                        tsdf_planner=tsdf_planner,
                        rgb_egocentric_views=rgb_egocentric_views,
                        cfg=cfg,
                        pts = pts,
                        verbose=True,
                    )
                    if vlm_response is None:
                        n_filtered_frames = 0
                        logging.info(
                            f"Episode id {episode_id} invalid: query_vlm_for_response failed!"
                        )
                        break

                    target_type, max_point_choice, n_filtered_frames, target_index = vlm_response
                    decision['target_type'] = target_type
                    decision['max_point_choice'] = target_index
                    # set the vlm choice as the navigation target
                    update_success = tsdf_planner.set_next_navigation_point(
                        target_type = target_type, 
                        choice=max_point_choice,
                        pts=pts,
                        objects=scene.objects,
                        obs_points = scene.all_obs_point,
                        cfg=cfg.planner,
                        pathfinder=scene.pathfinder,
                    )
                    if not update_success:
                        logging.info(
                            f"Subtask id {episode_id} invalid: set_next_navigation_point failed!"
                        )
                        break
                timer.stop("Querying the VLM")

                timer.start("Planner navigate to the target point")
                # (5) Agent navigate to the target point for one step
                return_values = tsdf_planner.agent_step(
                    pts=pts,
                    angle=angle,
                    pathfinder=scene.pathfinder,
                    cfg=cfg.planner,
                    path_points=None,
                    save_visualization=cfg.save_visualization,
                )
                if return_values[0] is None:
                    logging.info(
                        f"Subtask id {episode_id} invalid: agent_step failed!"
                    )
                    break

                # update agent's position and rotation
                pts, angle, pts_voxel, fig, _, target_arrived = return_values
                
                
                logger.log_step(pts_voxel=pts_voxel)
                logging.info(
                    f"Current position: {pts}, {logger.subtask_explore_dist:.3f}"
                )

                # sanity check about objects, scene graph, ...
                scene.sanity_check(cfg=cfg)

                if cfg.save_visualization:
                    # save the top-down visualization
                    logger.save_topdown_visualization(
                        global_step=cnt_step,
                        episode_id=episode_id,
                        subtask_metadata=subtask_metadata,
                        goal_obj_ids_mapping=goal_obj_ids_mapping,
                        fig=fig,
                    )
                    # save the visualization of vlm's choice at each step
                    logger.save_frontier_visualization(
                        global_step=cnt_step,
                        episode_id=episode_id,
                        tsdf_planner=tsdf_planner,
                        max_point_choice=max_point_choice,
                        global_caption=f"{subtask_metadata['question']}\n{subtask_metadata['task_type']}\n{subtask_metadata['class']}",
                    )
                timer.stop("Planner navigate to the target point")

                timer.print_summary(logging)
                # (6) Check if the agent has arrived at the target to finish the question
                # Final verification: use VLM to confirm the reached target
                if target_type != "frontier" and target_arrived:
                    all_frames = {}
                    back_frames = min(cnt_step + 1, cfg.frames_to_check)
                    
                    for back_step in range(cnt_step, cnt_step - back_frames, -1):
                        task_check_obs = []
                        for view_idx, ang in enumerate(all_angles):
                            obs_file_name = f"{back_step}-view_{view_idx}.png"
                            task_check_obs.append(
                                resize_image(scene.all_observations[obs_file_name], cfg.prompt_h, cfg.prompt_w)
                            )
                        all_frames[cnt_step - back_step + 1] = task_check_obs.copy()
                    vlm_response = query_vlm_for_response_end(
                        subtask_metadata=subtask_metadata,
                        rgb_egocentric_views=all_frames,
                        cfg=cfg,
                        verbose=True,
                    ) 
                    if (vlm_response == 'yes'):
                        break
                    else:
                        decision['object_judge'] = "no"

                his_decision[cnt_step] = decision

            # calculate the distance to the nearest view point
            agent_subtask_distance = calc_agent_subtask_distance(
                pts, subtask_metadata["viewpoints"], scene.pathfinder
            )
            if agent_subtask_distance < cfg.success_distance:
                success_by_distance = True
                logging.info(
                    f"Success: agent reached the target viewpoint at distance {agent_subtask_distance}!"
                )
            else:
                success_by_distance = False
                logging.info(
                    f"Fail: agent failed to reach the target viewpoint at distance {agent_subtask_distance}!"
                )

            logger.log_subtask_result(
                success_by_distance=success_by_distance,
                scene_id = scene_id,
                episode_id=episode_id,
                gt_subtask_explore_dist=subtask_metadata["gt_subtask_explore_dist"],
                goal_type=goal_types[0],
                n_filtered_frames=n_filtered_frames,
                n_total_frames=len(scene.img_to_edge),
            )

            logging.info(f"Scene graph of question {episode_id}:")
            logging.info(f"Question: {subtask_metadata['question']}")
            logging.info(f"Task type: {subtask_metadata['task_type']}")
            logging.info(f"Answer: {subtask_metadata['class']}")
            #scene.print_scene_graph()

            if not cfg.save_visualization:
                # clear up the stored images to save memory
                os.system(
                    f"rm -r {os.path.join(str(cfg.output_dir), f'{episode_id}')}"
                )

            if global_episode % 10 == 0:
                # save the results at the end of each episode
                logger.save_results()
                logging.info(f"Save Result at {global_episode}")

            logging.info(f"Episode {global_episode} (ID: {episode_id}) finish")
            if not cfg.save_visualization:
                os.system(f"rm -r {episode_dir}")

    logger.save_results()
    # aggregate the results from different splits into a single file
    logger.aggregate_results()

    logging.info(f"All scenes finish")


if __name__ == "__main__":
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    parser.add_argument("--specific", help="specific scene when multi-gpu evaluation", default=None, type=int)
    parser.add_argument("-ag", "--aggregate", action='store_true', default=False)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    
    # Set up logging
    
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    if args.specific is not None:
        logging_path = os.path.join(
            str(cfg.output_dir),
            f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}_{args.specific}.log",
        )
    else:
        logging_path = os.path.join(
            str(cfg.output_dir),
            f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}.log",
        )
    os.system(f"cp {args.cfg_file} {cfg.output_dir}")

    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    # Set up the logging format
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Set the custom formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    if args.aggregate:
        args.specific = -1
        logger = Logger(
            cfg.output_dir, args.start_ratio, args.end_ratio, voxel_size=cfg.tsdf_grid_size, specific=args.specific
        )
        logger.aggregate_results()
        exit(0)
    main(cfg, start_ratio=args.start_ratio, end_ratio=args.end_ratio, specific=args.specific)
