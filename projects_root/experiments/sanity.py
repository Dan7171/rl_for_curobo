import os
path = '/home/evrond/analysis_data/final_analysis/0709/500ts/'

def profile_dirname(dirname):
    split_dir = dirname.split('_')
    level = split_dir[-1]
    seed = split_dir[-2]
    task = split_dir[-3]
    alg = split_dir[-4]
    timestamp = dirname[:19]
    return {'level': level, 'seed': seed, 'task': task, 'alg': alg,'id': dirname,'timestamp': timestamp}

def make_dir_list(path):
    dir_list = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if 'ur5e' in dir:
                dir_list.append(dir)
    return dir_list

def get_profiled_dirs(dir_list):
    filtered = []
    for dir in dir_list:
        filtered.append((profile_dirname(dir), dir))
    return filtered

def check_unique(filtered):
    unique_filtered = []
    duplicates_filtered = []
    unique = []
    duplicates = []
    for f in filtered:
        filtered_dirname, dirname = f
        if filtered_dirname not in unique_filtered:
            unique_filtered.append(filtered_dirname)
            unique.append(dirname)
        else:
            duplicates.append(dirname)
            duplicates_filtered.append(filtered_dirname)

    return unique, duplicates, unique_filtered, duplicates_filtered
    


def find_id_duplicates(filtered):
    found_ids = set()
    # found_filtered = set()
    timestamp_to_filtered = {}
    for f in filtered:
        filtered_dirname, dirname = f
        if filtered_dirname['timestamp'] not in timestamp_to_filtered:
            timestamp_to_filtered[filtered_dirname['timestamp']] = []
        timestamp_to_filtered[filtered_dirname['timestamp']].append(filtered_dirname)
    
    duplicates_id_to_filtered = {}
    for k, v in timestamp_to_filtered.items():
        if len(v) > 1:
            duplicates_id_to_filtered[k] = v
    return duplicates_id_to_filtered
filtered = get_profiled_dirs(make_dir_list(path))
unique, duplicates, unique_filtered, duplicates_filtered = check_unique(filtered)
print(len(unique))
print(len(duplicates))
# print(duplicates)
for i in range(len(duplicates)):
    print(duplicates[i])
    print(duplicates_filtered[i])
duplicates_id_to_filtered = find_id_duplicates(filtered)
for k, v in duplicates_id_to_filtered.items():
    print(k)
    for v in v:
        print(v)



