import os
import re

# Configuration parameters
folder_path = '/home/dataset-assist-0/hsun/hsun/MSGNav/results/merge'  # Path to log files folder
new_thresholds = [0.25, 1.0]
for new_threshold in new_thresholds:
    # new_threshold = 0.5  # Alternative distance threshold example

    # Regular expressions
    success_fail_pattern = r'\d{2}:\d{2}:\d{2} - (Success|Fail): .*? at distance (\d+\.\d+|\d+\.\d+e[+-]?\d+|\d+)!'
    spl_pattern = r'\d{2}:\d{2}:\d{2} - SPL by distance: (\d+\.\d+)'

    # Initialize statistics
    original_success_distances = []
    still_success_count = 0
    total_original_success = 0
    count = 0
    # Store SPL mean and back-computed SPL values
    spl = 0
    spl_raw = 0
    bfspl = 0
    # Iterate over all .log files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.log'):
            cnt = 1
            bfspl = 0
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    # Parse Success/Fail records
                    sf_match = re.search(success_fail_pattern, line)
                    if sf_match:
                        record_type, distance_str = sf_match.groups()
                        try:
                            distance = float(distance_str)
                            if record_type == 'Success':
                                total_original_success += 1
                                original_success_distances.append(distance)
                            if distance <= new_threshold:
                                still_success_count += 1
                            count+=1
                        except ValueError:
                            print(f"Failed to parse distance: {distance_str} in line: {line.strip()}")

                    # Parse SPL mean
                    spl_match = re.search(spl_pattern, line)
                    if spl_match:
                        spl_mean = float(spl_match.group(1))
                        ans = spl_mean * cnt - (cnt-1) * bfspl
                        bfspl = spl_mean
                        cnt += 1
                        spl_raw += ans
                        spl += ans if distance <= new_threshold else 0

    # Print results
    print(f"===== Success statistics (count: {total_original_success}/{count} → {still_success_count}/{count}) =====")
    print(f"Under threshold {new_threshold}: ")
    if total_original_success > 0:
        ratio = still_success_count / count * 100
        print(f"Ratio: {ratio:.1f}%")
        print(f"New global average SPL: {spl/count:.1f}")