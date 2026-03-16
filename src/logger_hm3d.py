import os
import json
import pickle
from collections import defaultdict
import logging
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image
from typing import Union
import math
import habitat_sim

from src.multimodal_3d_scene_graph import Scene
from src.tsdf_planner import TSDFPlanner, Frontier


class Logger:
    def __init__(
        self,
        output_dir,
        start_ratio,
        end_ratio,
        voxel_size,  # used for calculating the moving distance
        specific=None,  # used for specific scene
    ):
        self.output_dir = output_dir
        self.voxel_size = voxel_size
        if specific is None:
            self.success_by_distance_path = success_by_distance_path = os.path.join(
                output_dir, f"success_by_distance_{start_ratio}_{end_ratio}.pkl"
            )
            self.spl_by_distance_path = spl_by_distance_path = os.path.join(
                output_dir, f"spl_by_distance_{start_ratio}_{end_ratio}.pkl"
            )
            self.success_by_task_path = success_by_task_path = os.path.join(
                output_dir, f"success_by_task_{start_ratio}_{end_ratio}.pkl"
            )
            self.spl_by_task_path = spl_by_task_path = os.path.join(
                output_dir, f"spl_by_task_{start_ratio}_{end_ratio}.pkl"
            )
            self.n_filtered_frames_path = n_filtered_frames_path = os.path.join(
                output_dir,
                f"n_filtered_frames_{start_ratio}_{end_ratio}.json",
            )
            self.n_total_frames_path = n_total_frames_path = os.path.join(
                output_dir, f"n_total_frames_{start_ratio}_{end_ratio}.json"
            )       
        else:
            self.success_by_distance_path = success_by_distance_path = os.path.join(
                output_dir, f"success_by_distance_{start_ratio}_{end_ratio}_{specific}.pkl"
            )   
            self.spl_by_distance_path = spl_by_distance_path = os.path.join(
                output_dir, f"spl_by_distance_{start_ratio}_{end_ratio}_{specific}.pkl"
            )
            self.success_by_task_path = success_by_task_path = os.path.join(
                output_dir, f"success_by_task_{start_ratio}_{end_ratio}_{specific}.pkl"
            )
            self.spl_by_task_path = spl_by_task_path = os.path.join(
                output_dir, f"spl_by_task_{start_ratio}_{end_ratio}_{specific}.pkl"
            )
            self.n_filtered_frames_path = n_filtered_frames_path = os.path.join(
                output_dir,
                f"n_filtered_frames_{start_ratio}_{end_ratio}_{specific}.json",
            )
            self.n_total_frames_path = n_total_frames_path = os.path.join(
                output_dir, f"n_total_frames_{start_ratio}_{end_ratio}_{specific}.json"
            )
        if os.path.exists(
            success_by_distance_path
        ):
            self.success_by_distance = pickle.load(
                open(
                    success_by_distance_path,
                    "rb",
                )
            )
        else:
            self.success_by_distance = {}  # subtask id -> success
        
        if os.path.exists(
            spl_by_distance_path
        ):
            self.spl_by_distance = pickle.load(
                open(
                    spl_by_distance_path,
                    "rb",
                )
            )
        else:
            self.spl_by_distance = {}  # subtask id -> spl
        if os.path.exists(
            success_by_task_path
        ):
            self.success_by_task = pickle.load(
                open(
                    success_by_task_path,
                    "rb",
                )
            )
        else:
            # success_by_task = {}  # task type -> success
            self.success_by_task = defaultdict(list)
        if os.path.exists(
            spl_by_task_path
        ):
            self.spl_by_task = pickle.load(
                open(
                    spl_by_task_path,
                    "rb",
                )
            )
        else:
            # spl_by_task = {}  # task type -> spl
            self.spl_by_task = defaultdict(list)
        
        if os.path.exists(
            n_filtered_frames_path
        ):
            with open(
                n_filtered_frames_path,
                "r",
            ) as f:
                self.n_filtered_frames_list = json.load(f)
        else:
            self.n_filtered_frames_list = {}
        
        if os.path.exists(
            n_total_frames_path
        ):
            with open(
                n_total_frames_path,
                "r",
            ) as f:
                self.n_total_frames_list = json.load(f)
        else:
            self.n_total_frames_list = {}

        self.start_ratio = start_ratio
        self.end_ratio = end_ratio

        # some sanity check
        assert (
            len(self.success_by_distance)
            == len(self.spl_by_distance)
        ), f"{len(self.success_by_distance)} != {len(self.spl_by_distance)}"
        assert (
            sum([len(task_res) for task_res in self.success_by_task.values()])
            == sum([len(task_res) for task_res in self.spl_by_task.values()])
        ), f"{sum([len(task_res) for task_res in self.success_by_task.values()])} != {sum([len(task_res) for task_res in self.spl_by_task.values()])}"

        # logging for episode
        self.episode_dir = None

        # logging for subtask
        self.subtask_object_observe_dir = None
        self.pts_voxels = np.empty((0, 2))
        self.subtask_explore_dist = 0.0

    def save_results(self):
        # some sanity check
        assert (
            len(self.success_by_distance)
            == len(self.spl_by_distance)
        ), f"{len(self.success_by_distance)} != {len(self.spl_by_distance)}"
        assert (
            sum([len(task_res) for task_res in self.success_by_task.values()])
            == sum([len(task_res) for task_res in self.spl_by_task.values()])
        ), f"{sum([len(task_res) for task_res in self.success_by_task.values()])} != {sum([len(task_res) for task_res in self.spl_by_task.values()])}"

        
        with open(
            self.success_by_distance_path,
            "wb",
        ) as f:
            pickle.dump(self.success_by_distance, f)
    
        with open(
            self.spl_by_distance_path,
            "wb",
        ) as f:
            pickle.dump(self.spl_by_distance, f)
        
        with open(
            self.success_by_task_path,
            "wb",
        ) as f:
            pickle.dump(self.success_by_task, f)
        with open(
            self.spl_by_task_path,
            "wb",
        ) as f:
            pickle.dump(self.spl_by_task, f)
        with open(
            self.n_filtered_frames_path,
            "w",
        ) as f:
            json.dump(self.n_filtered_frames_list, f, indent=4)
        
        with open(
            self.n_total_frames_path,
            "w",
        ) as f:
            json.dump(self.n_total_frames_list, f, indent=4)

    def aggregate_results(self):
        # aggregate the results into a single file
        filenames_to_merge = [
            "success_by_distance",
            "spl_by_distance",
        ]
        for filename in filenames_to_merge:
            all_results = {}
            all_results_paths = glob.glob(
                os.path.join(self.output_dir, f"{filename}_*.pkl")
            )
            for results_path in all_results_paths:
                with open(results_path, "rb") as f:
                    all_results.update(pickle.load(f))
            logging.info(
                f"Total {filename} results: {100 * np.mean(list(all_results.values())):.2f}, len: {len(all_results)}"
            )
            with open(os.path.join(self.output_dir, f"{filename}.pkl"), "wb") as f:
                pickle.dump(all_results, f)
        filenames_to_merge = ["success_by_task", "spl_by_task"]
        for filename in filenames_to_merge:
            all_results = {}
            all_results_paths = glob.glob(
                os.path.join(self.output_dir, f"{filename}_*.pkl")
            )
            for results_path in all_results_paths:
                with open(results_path, "rb") as f:
                    separate_stat = pickle.load(f)
                    for task_name, task_res in separate_stat.items():
                        if task_name not in all_results:
                            all_results[task_name] = []
                        all_results[task_name] += task_res
            for task_name, task_res in all_results.items():
                logging.info(
                    f"Total {filename} results for {task_name}: {100 * np.mean(task_res):.2f}, len: {len(task_res)}"
                )
            with open(os.path.join(self.output_dir, f"{filename}.pkl"), "wb") as f:
                pickle.dump(all_results, f)

        n_filtered_frames_list = {}
        all_n_filtered_frames_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_filtered_frames_*.json")
        )
        for n_filtered_frames_list_path in all_n_filtered_frames_list_paths:
            with open(n_filtered_frames_list_path, "r") as f:
                n_filtered_frames_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_filtered_frames.json"), "w") as f:
            json.dump(n_filtered_frames_list, f, indent=4)
        logging.info(
            f"Average number of filtered frames: {np.mean(list(n_filtered_frames_list.values()))}"
        )

        n_total_frames_list = {}
        all_n_total_frames_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_total_frames_*.json")
        )
        for n_total_frames_list_path in all_n_total_frames_list_paths:
            with open(n_total_frames_list_path, "r") as f:
                n_total_frames_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_total_frames.json"), "w") as f:
            json.dump(n_total_frames_list, f, indent=4)
        logging.info(
            f"Average number of total frames: {np.mean(list(n_total_frames_list.values()))}"
        )

    def log_subtask_result(
        self,
        success_by_distance: bool,
        scene_id,
        episode_id: str,
        gt_subtask_explore_dist: float,
        goal_type: str,
        n_filtered_frames,
        n_total_frames,
    ):
        key_id = str(scene_id)+'_'+str(episode_id)
        if success_by_distance:
            self.success_by_distance[key_id] = 1.0
        else:
            self.success_by_distance[key_id] = 0.0


        self.spl_by_distance[key_id] = (
            self.success_by_distance[key_id]
            * gt_subtask_explore_dist
            / max(gt_subtask_explore_dist, self.subtask_explore_dist)
        )

        if math.isnan(self.spl_by_distance[key_id]):
            self.spl_by_distance[key_id] = 0
        self.success_by_task[goal_type].append(self.success_by_distance[key_id])
        self.spl_by_task[goal_type].append(self.spl_by_distance[key_id])

        logging.info(
            f"Subtask {key_id} finished, {self.subtask_explore_dist} length"
        )
        logging.info(
            f"Success rate by distance: {100 * np.mean(np.asarray(list(self.success_by_distance.values()))):.2f}"
        )
        logging.info(
            f"SPL by distance: {100 * np.mean(np.asarray(list(self.spl_by_distance.values()))):.2f}"
        )

        for task_name, success_list in self.success_by_task.items():
            logging.info(
                f"Success rate for {task_name}: {100 * np.mean(np.asarray(success_list)):.2f}"
            )
        for task_name, spl_list in self.spl_by_task.items():
            logging.info(
                f"SPL for {task_name}: {100 * np.mean(np.asarray(spl_list)):.2f}"
            )

        logging.info(
            f"Filtered frames/Total frames: {n_filtered_frames}/{n_total_frames}"
        )
        # save the number of frames
        self.n_filtered_frames_list[key_id] = n_filtered_frames
        self.n_total_frames_list[key_id] = n_total_frames

        # clear the subtask logging
        self.subtask_object_observe_dir = None
        self.pts_voxels = np.empty((0, 2))
        self.subtask_explore_dist = 0.0

    def init_episode(
        self,
        episode_id,
    ):
        self.episode_dir = os.path.join(self.output_dir, episode_id)
        eps_frontier_dir = os.path.join(self.episode_dir, "frontier")

        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(eps_frontier_dir, exist_ok=True)

        return self.episode_dir, eps_frontier_dir

    def init_subtask(
        self,
        object_category,
        episode_id,
        goals,
        pts,
        scene: Scene,  # used to get image goal observation and use its pathfinder
        tsdf_planner: TSDFPlanner,
    ):
        # determine the navigation goals
        
        goal_obj_ids = [x["object_id"] for x in goals]

        goal_positions = [x["position"] for x in goals]
        goal_positions_voxel = [tsdf_planner.habitat2voxel(p) for p in goal_positions]

        viewpoints = [
            view_point["agent_state"]["position"]
            for goal in goals
            for view_point in goal["view_points"]
        ]
        # get the shortest distance from current position to the viewpoints
        all_distances = []
        for viewpoint in viewpoints:
            path = habitat_sim.ShortestPath()
            path.requested_start = pts
            path.requested_end = viewpoint
            found_path = scene.pathfinder.find_path(path)
            if not found_path:
                all_distances.append(np.inf)
            else:
                all_distances.append(path.geodesic_distance)
        gt_subtask_explore_dist = min(all_distances) + 1e-6

        self.subtask_object_observe_dir = os.path.join(
            self.output_dir, episode_id, "object_observations"
        )
        if os.path.exists(self.subtask_object_observe_dir):
            os.system(f"rm -r {self.subtask_object_observe_dir}")
        os.makedirs(self.subtask_object_observe_dir, exist_ok=False)

        # Prepare metadata for the subtask
        subtask_metadata = {
            "question_id": None,
            "question": None,
            "image": None,
            "answer": object_category,
            "goal_obj_ids": goal_obj_ids,  # this is a list of obj id, since for object class type, there will be multiple target objects
            "class": object_category,
            "goal_positions_voxel": goal_positions_voxel,  # also a list of positions for possible multiple objects
            "task_type": 'object',
            "viewpoints": viewpoints,
            "gt_subtask_explore_dist": gt_subtask_explore_dist,
        }
        subtask_metadata["question"] = f"Can you find the {object_category}?"

        self.pts_voxels = np.empty((0, 2))
        self.pts_voxels = np.vstack(
            [self.pts_voxels, tsdf_planner.habitat2voxel(pts)[:2]]
        )
        self.subtask_explore_dist = 0.0

        return subtask_metadata

    def log_step(self, pts_voxel):
        self.pts_voxels = np.vstack([self.pts_voxels, pts_voxel])
        self.subtask_explore_dist += (
            np.linalg.norm(self.pts_voxels[-1] - self.pts_voxels[-2]) * self.voxel_size
        )

    def save_topdown_visualization(
        self, global_step, episode_id, subtask_metadata, goal_obj_ids_mapping, fig
    ):
        assert self.episode_dir is not None
        visualization_path = os.path.join(self.episode_dir, "visualization")
        os.makedirs(visualization_path, exist_ok=True)

        ax1 = fig.axes[0]
        ax1.plot(
            self.pts_voxels[:-1, 1], self.pts_voxels[:-1, 0], linewidth=1, color="white"
        )
        ax1.scatter(self.pts_voxels[0, 1], self.pts_voxels[0, 0], c="white", s=50)

        # add target object bbox
        for goal_id, goal_pos_voxel in zip(
            subtask_metadata["goal_obj_ids"], subtask_metadata["goal_positions_voxel"]
        ):
            color = "green" if len(goal_obj_ids_mapping[goal_id]) > 0 else "red"
            ax1.scatter(goal_pos_voxel[1], goal_pos_voxel[0], c=color, s=120)

        fig.tight_layout()
        plt.savefig(os.path.join(visualization_path, f"{global_step}_{episode_id}.png"))
        plt.close()

    def save_frontier_visualization(
        self,
        global_step,
        episode_id,
        tsdf_planner: TSDFPlanner,
        max_point_choice,
        global_caption,
    ):
        assert self.episode_dir is not None
        frontier_video_path = os.path.join(self.episode_dir, "frontier_video")
        episode_frontier_dir = os.path.join(self.episode_dir, "frontier")
        os.makedirs(frontier_video_path, exist_ok=True)
        num_images = len(tsdf_planner.frontiers)
        side_length = int(np.sqrt(num_images)) + 1
        side_length = max(2, side_length)
        fig, axs = plt.subplots(side_length, side_length, figsize=(20, 20))
        for h_idx in range(side_length):
            for w_idx in range(side_length):
                axs[h_idx, w_idx].axis("off")
                i = h_idx * side_length + w_idx
                if (i < num_images - 1) or (
                    i < num_images and type(max_point_choice) == Frontier
                ):
                    img_path = os.path.join(
                        episode_frontier_dir, tsdf_planner.frontiers[i].image
                    )
                    img = matplotlib.image.imread(img_path)
                    axs[h_idx, w_idx].imshow(img)
                    if (
                        type(max_point_choice) == Frontier
                        and max_point_choice.image == tsdf_planner.frontiers[i].image
                    ):
                        axs[h_idx, w_idx].set_title("Chosen")
        fig.suptitle(global_caption, fontsize=16)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        plt.savefig(
            os.path.join(frontier_video_path, f"{global_step}_{episode_id}.png")
        )
        plt.close()
