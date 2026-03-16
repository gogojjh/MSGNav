import os
import sys  
  
def main(scene_id, device_id, start_ratio=0.0, end_ratio=1.0, split=1):  
    print(f"Infer Scene {scene_id} with episode {split} on Device {device_id}") 
    os.system(f'CUDA_VISIBLE_DEVICES={device_id} python run_goatbench_evaluation.py -cf cfg/eval_goatbench.yaml --start_ratio {start_ratio} --end_ratio {end_ratio} --specific {scene_id} --split {split}') 
    
if __name__ == "__main__":  
    if len(sys.argv) > 1:  
        scene_id = int(sys.argv[1])  
        device_id = int(sys.argv[2])
        start_ratio = float(sys.argv[3])
        end_ratio = float(sys.argv[4])
        split = int(sys.argv[5])
        main(scene_id, device_id, start_ratio, end_ratio, split)  

    print('done')
