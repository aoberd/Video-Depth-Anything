import os
import subprocess
import multiprocessing
import random
import time


def list_deepest_directories(root_dir):
    deepest_dirs = []
    max_depth = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        current_depth = dirpath.count(os.sep) - root_dir.count(os.sep)
        
        if current_depth > max_depth:
            max_depth = current_depth
            deepest_dirs = [dirpath]
        elif current_depth == max_depth:
            deepest_dirs.append(dirpath)

    return deepest_dirs


def split_and_save(deepest_dirs, num_gpus):
    chunk_size = len(deepest_dirs) // num_gpus
    file_names = []
    for i in range(num_gpus):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size if i < num_gpus - 1 else len(deepest_dirs)
        part = deepest_dirs[start_index:end_index]
        
        file_path = f'/tmp/vda_{i}.txt'
        with open(file_path, 'w') as file:
            for dir in part:
                file.write(f"{dir}\n")

        file_names.append(file_path)

    return file_names


def create_depth_from_directory(X, fps=24, input_extension=".jpg", output_extension=".jpg"):
    dirpath = X[0]
    GPU = X[1]
    # time.sleep( 4*(X[1] + 8*X[2]) )

    print(f"Processing: {dirpath}")

    command = [
        "python3",
        "run.py",
        "--device", "cuda:"+str(GPU),
        "--input", dirpath,
        # "--output_dir", "/home/aobe/Video-Depth-Anything/outputs",
        "--encoder", "vitl",
        "--save_jpeg",
        "--delete_originals",
    ]

    subprocess.run(command, cwd="/home/aobe/Video-Depth-Anything/", check=True, capture_output=True, text=True)


def create_depth_from_tree(root_dir, num_gpus=8, num_parallel=2, **kwargs):
    """
    Processes a directory tree and creates a video from images in each folder.

    Args:
        root_dir (str): The root directory to start processing.
        fps (int): The frames per second for the generated videos (default: 24).
        image_extension (str): The file extension of the images to include (default: ".jpg").
    """

    dirs = list_deepest_directories(root_dir)
    filelists = split_and_save(dirs, num_gpus*num_parallel)

    N = 0
    X = []
    for listfile in filelists:
        N = (N + 1) 
        X.append((listfile, N % num_gpus, N // num_gpus))

    with multiprocessing.Pool(processes=num_gpus*num_parallel) as pool:
        pool.starmap(create_depth_from_directory, [(x,) + (tuple(kwargs.items()),) for x in X])
 

if __name__ == "__main__":
    root_directory = '/home/aobe/scratch/parallelTest/'

    create_depth_from_tree(root_directory, num_gpus=8, num_parallel=1, fps=24, input_extension="-bayerRG8.jpg", output_extension=".jpg") 
    print("Processing complete.")


##################################
# leftover stuff 
##################################

# def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2, wait=False, sleep_time=10):
#     """
#     Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
#     This function should be called after all imports,
#     in case you are setting CUDA_AVAILABLE_DEVICES elsewhere
#     Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
#     Args:
#         threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
#                                               Defaults to 1500 (MiB).
#         max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
#                                   Defaults to 2.
#         wait (bool, optional): Whether to wait until a GPU is free. Default False.
#         sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
#     """

#     def _check():
#         # Get the list of GPUs via nvidia-smi
#         smi_query_result = subprocess.check_output(
#             "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
#         )
#         # Extract the usage information
#         gpu_info = smi_query_result.decode("utf-8").split("\n")
#         gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
#         gpu_info = [
#             int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
#         ]  # Remove garbage
#         # Keep gpus under threshold only
#         free_gpus = [
#             str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
#         ]
#         free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
#         gpus_to_use = ",".join(free_gpus)
#         return gpus_to_use

#     while True:
#         gpus_to_use = _check()
#         if gpus_to_use or not wait:
#             break
#         print(f"No free GPUs found, retrying in {sleep_time}s")
#         time.sleep(sleep_time)
    
#     return gpus_to_use

    # N = 0
    # X = []
    # base_depth = root_dir.rstrip(os.path.sep).count(os.path.sep)
    # for dirpath, dirnames, filenames in os.walk(root_dir):
        # cur_depth = dirpath.count(os.path.sep)
        # if cur_depth-base_depth != 2:
        #     continue

        # N = (N + 1) % num_gpus
        # X.append((dirpath, N))

    # command = [
    #     "ffmpeg",
    #     "-i", depth_video,
    #     "-q:v", "1",
    #     "-y", # Overwrite output file if it exists
    #     output_pattern
    # ]

    # subprocess.run(command, check=True, capture_output=True, text=True)        


    # for dirpath, dirnames, filenames in os.walk(dirpath):
    #     image_files = [f for f in filenames if f.endswith(input_extension)]

    # Construct the input pattern for ffmpeg
    # input_pattern = os.path.join(dirpath, "*" + input_extension)  
    # output_pattern = os.path.join(dirpath, "depth%05d" + output_extension)  
    
    # Output video file name
    # rgb_video = os.path.join(dirpath, "rgb.mp4")
    # depth_video = os.path.join(dirpath, "rgb_vis.mp4")

    # command = [
    #     "ffmpeg",
    #     "-pattern_type", "glob",
    #     "-i", input_pattern,
    #     "-c:v", "libx264",
    #     "-preset", "ultrafast",
    #     "-crf", "17",
    #     "-y",  # Overwrite output file if it exists
    #     rgb_video
    # ]

    # subprocess.run(command, check=True, capture_output=True, text=True)

    # for f in image_files:
    #     os.remove(os.path.join(dirpath, f))

    # time.sleep(5*random.random())
    # free_GPUs = assign_free_gpus(threshold_vram_usage=20000, max_gpus=8, wait=True, sleep_time=10)