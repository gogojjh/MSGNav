import multiprocessing
import subprocess
from multiprocessing import Manager

def dynamic_scene_process(process_id, device_id, total_processes, start_ratio = 0.0, end_ratio = 1.0):  
    while not task_queue.empty():
        task_id = task_queue.get(block=True)
        t = subprocess.Popen(
            ["python", "-m", "start_hm3d", str(task_id % 36), str(device_id), str(start_ratio), str(end_ratio), str(int(task_id / 36) + 1)],
        )
        t.wait()

if __name__ == "__main__":  
    #devices = [0,1,2,7]
    devices = [0,1,2,3,4,5,6]
    total_episodes = 36
    processes = []
    
    task_queue = multiprocessing.Queue()
    
    splits = 1
    [task_queue.put(i) for i in range(0, total_episodes * splits)]
    pool = multiprocessing.Pool(processes = len(devices))
    for i in range(len(devices)):  
        pool.apply_async(dynamic_scene_process, (i, devices[i], len(devices)))
    
    
    pool.close()
    pool.join()

    t = subprocess.Popen(
            ["python", "-m", "start_hm3d", str(-1), str(devices[0]), str(0.0), str(1.0), str(1)],
        )
    t.wait()