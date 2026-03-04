import os
import sys  
  
def main(scene_id, device_id, start_ratio=0.0, end_ratio=1.0):  
    print(f"Infer Scene {scene_id} on Device {device_id}")  
    os.system(f'CUDA_VISIBLE_DEVICES={device_id} python run_hm3d_evaluation.py -cf cfg/eval_hm3d.yaml --start_ratio {start_ratio} --end_ratio {end_ratio} --specific {scene_id}') 
    
if __name__ == "__main__":  
    if len(sys.argv) > 1:  
        scene_id = int(sys.argv[1])  
        device_id = int(sys.argv[2])
        start_ratio = float(sys.argv[3])
        end_ratio = float(sys.argv[4])
        main(scene_id, device_id, start_ratio, end_ratio)  

    print('done')
