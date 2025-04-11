# This file is originally from DepthCrafter/depthcrafter/utils.py at main · Tencent/DepthCrafter
# SPDX-License-Identifier: MIT License license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file is released under [ MIT License license], with the full license text available at [https://github.com/Tencent/DepthCrafter?tab=License-1-ov-file].
import numpy as np
import matplotlib.cm as cm
from pathlib import Path
import imageio
import cv2
import os
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except:
    DECORD_AVAILABLE = False


def ensure_even(value):
    return value if value % 2 == 0 else value + 1

def resize_and_crop_to_smallest(frames):
    # Find the smallest dimensions
    min_height = min(frame.shape[0] for frame in frames)
    min_width = min(frame.shape[1] for frame in frames)

    resized_frames = []
    for frame in frames:
        # Resize proportionally to the smallest image
        ratio = min(min_height / frame.shape[0], min_width / frame.shape[1])
        new_height = ensure_even(round(frame.shape[0] * ratio))
        new_width = ensure_even(round(frame.shape[1] * ratio))
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Crop the longer side
        if new_height > min_height:
            start_y = max(0, (new_height - min_height)) // 2
            cropped_frame = resized_frame[start_y:start_y + min_height, :]
        else:
            start_x = max(0, (new_width - min_width)) // 2
            cropped_frame = resized_frame[:, start_x:start_x + min_width]

        resized_frames.append(cropped_frame)

    return resized_frames


def read_video_frames(input_path, process_length, target_fps=-1, max_res=-1, is_image_sequence=False, delete_originals=False):
    """
    Reads frames from a video file or an image sequence.

    Args:
        input_path (str or Path): Path to the video file or image sequence directory.
        process_length (int): Maximum number of frames to process. -1 for all.
        target_fps (float, optional): Target frames per second. Defaults to -1 (original).
        max_res (int, optional): Maximum resolution (height or width). Defaults to -1 (no scaling).
        is_image_sequence (bool, optional):  If True, reads from an image sequence in a directory.
                                               Defaults to False (reads from video).

    Returns:
        tuple: A tuple containing the frames (numpy array) and the frames per second (float).
    """

    if is_image_sequence:
        input_path = Path(input_path)
        if not input_path.is_dir():
            raise ValueError("Invalid image sequence directory.")

        image_files = sorted([f for f in input_path.glob("*") if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        if not image_files:
            raise ValueError("No images found in the sequence directory.")

        frames = []
        for image_path in image_files:
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Warning: Could not read image {image_path}")
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            if max_res > 0 and max(frame.shape[0], frame.shape[1]) > max_res:
                scale = max_res / max(frame.shape[0], frame.shape[1])
                height = ensure_even(round(frame.shape[0] * scale))
                width = ensure_even(round(frame.shape[1] * scale))
                frame = cv2.resize(frame, (width, height))

            frames.append(frame)

        if process_length != -1 and process_length < len(frames):
            frames = frames[:process_length]

        frames = resize_and_crop_to_smallest(frames)

        frames = np.stack(frames, axis=0)
        fps = target_fps if target_fps > 0 else 30  # Default FPS for image sequence

        if delete_originals:
            for f in image_files:
                os.remove(os.path.join(input_path, f))        

    else:  # Video
        if DECORD_AVAILABLE:
            vid = VideoReader(input_path, ctx=cpu(0))
            original_height, original_width = vid.get_batch([0]).shape[1:3]
            height = original_height
            width = original_width
            if max_res > 0 and max(height, width) > max_res:
                scale = max_res / max(original_height, original_width)
                height = ensure_even(round(original_height * scale))
                width = ensure_even(round(original_width * scale))

            vid = VideoReader(input_path, ctx=cpu(0), width=width, height=height)

            fps = vid.get_avg_fps() if target_fps == -1 else target_fps
            stride = round(vid.get_avg_fps() / fps)
            stride = max(stride, 1)
            frames_idx = list(range(0, len(vid), stride))
            if process_length != -1 and process_length < len(frames_idx):
                frames_idx = frames_idx[:process_length]
            frames = vid.get_batch(frames_idx).asnumpy()
        else:
            cap = cv2.VideoCapture(input_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            if max_res > 0 and max(original_height, original_width) > max_res:
                scale = max_res / max(original_height, original_width)
                height = round(original_height * scale)
                width = round(original_width * scale)

            fps = original_fps if target_fps < 0 else target_fps

            stride = max(round(original_fps / fps), 1)

            frames = []
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (process_length > 0 and frame_count >= process_length):
                    break
                if frame_count % stride == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    if max_res > 0 and max(original_height, original_width) > max_res:
                        frame = cv2.resize(frame, (width, height))  # Resize frame
                    frames.append(frame)
                frame_count += 1
            cap.release()
            frames = np.stack(frames, axis=0)

        if delete_originals:
            os.remove(input_path)

    return frames, fps


def save_video(frames, output_video_path, fps=10, is_depths=False, grayscale=False):
    if output_video_path.endswith('.mp4'):
        writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
    else:
        from cv2 import imwrite

    if is_depths:
        colormap = np.array(cm.get_cmap("inferno").colors)
        d_min, d_max = frames.min(), frames.max()
        for i in range(frames.shape[0]):
            depth = frames[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm
            if output_video_path.endswith('.mp4'):
                writer.append_data(depth_vis)
            else:
                output_jpeg = f"{output_video_path}/depth_{i:05d}.jpg"
                imwrite(output_jpeg, cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
    else:
        for i in range(frames.shape[0]):
            if output_video_path.endswith('.mp4'):
                writer.append_data(frames[i])
            else:
                output_jpeg = f"{output_video_path}/image_{i:05d}.jpg"
                imwrite(output_jpeg, cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))

    if output_video_path.endswith('.mp4'):
        writer.close()
