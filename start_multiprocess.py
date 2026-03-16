import argparse
import multiprocessing
import queue
import subprocess
import sys


def parse_devices(devices_arg: str):
    return [int(device.strip()) for device in devices_arg.split(",") if device.strip()]


def dynamic_scene_process(
    worker_id,
    device_id,
    total_scenes,
    task_type,
    hm3d_module,
    goatbench_module,
    task_queue,
    start_ratio=0.0,
    end_ratio=1.0,
):
    print(f"[worker-{worker_id}] start on cuda:{device_id}", flush=True)
    while True:
        try:
            task_id = task_queue.get_nowait()
        except queue.Empty:
            break

        scene_id = task_id % total_scenes
        split_id = int(task_id / total_scenes) + 1
        if task_type == "hm3d":
            cmd = [
                sys.executable,
                "-m",
                hm3d_module,
                str(scene_id),
                str(device_id),
                str(start_ratio),
                str(end_ratio),
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                goatbench_module,
                str(scene_id),
                str(device_id),
                str(start_ratio),
                str(end_ratio),
                str(split_id),
            ]
        print(
            f"[worker-{worker_id}] run task_id={task_id}, scene={scene_id}, split={split_id}, device={device_id}",
            flush=True,
        )
        subprocess.run(cmd, check=False)

    print(f"[worker-{worker_id}] finished", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HM3D/GOAT-Bench tasks in parallel.")
    parser.add_argument(
        "--task",
        type=str,
        choices=["hm3d", "goatbench"],
        default="hm3d",
        help="Task type for parallel execution.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0,1,2,3,4,5,6",
        help="Comma-separated GPU ids, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--total_scenes",
        type=int,
        default=36,
        help="Number of scene tasks per split.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=1,
        help="Number of splits to run.",
    )
    parser.add_argument(
        "--start_ratio",
        type=float,
        default=0.0,
        help="Start ratio passed to task start script.",
    )
    parser.add_argument(
        "--end_ratio",
        type=float,
        default=1.0,
        help="End ratio passed to task start script.",
    )
    parser.add_argument(
        "--hm3d_module",
        type=str,
        default="start_hm3d",
        help="Python module name used when --task hm3d.",
    )
    parser.add_argument(
        "--goatbench_module",
        type=str,
        default="start_goatbench",
        help="Python module name used when --task goatbench.",
    )
    args = parser.parse_args()

    devices = parse_devices(args.devices)
    if len(devices) == 0:
        raise ValueError("No valid GPU id provided in --devices.")

    total_scenes = args.total_scenes
    splits = args.splits

    task_queue = multiprocessing.Queue()
    for i in range(total_scenes * splits):
        task_queue.put(i)

    workers = []
    for worker_id, device_id in enumerate(devices):
        p = multiprocessing.Process(
            target=dynamic_scene_process,
            args=(
                worker_id,
                device_id,
                total_scenes,
                args.task,
                args.hm3d_module,
                args.goatbench_module,
                task_queue,
                args.start_ratio,
                args.end_ratio,
            ),
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    # Final aggregation pass for hm3d (keeps original behavior).
    if args.task == "hm3d":
        final_cmd = [
            sys.executable,
            "-m",
            args.hm3d_module,
            str(-1),
            str(devices[0]),
            str(0.0),
            str(1.0),
        ]
        subprocess.run(final_cmd, check=False)