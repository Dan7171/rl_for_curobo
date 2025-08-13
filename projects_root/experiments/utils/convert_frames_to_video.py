#!/usr/bin/env python3
"""
Convert recorded frames to video using OpenCV or imageio
"""

import cv2
import imageio
import argparse
import os
import glob
from pathlib import Path
import subprocess


def frames_to_video_opencv(input_dir, output_video, fps=30):
    """Convert frames to video using OpenCV"""
    # Try different frame patterns to handle various directory structures
    frame_patterns = [
        os.path.join(input_dir, "rgb", "rgb_*.png"),  # Standard Isaac Sim structure
        os.path.join(input_dir, "rgb_*.png"),         # Direct directory structure
        os.path.join(input_dir, "*.png"),             # Any PNG files
    ]
    
    frame_files = []
    for pattern in frame_patterns:
        frame_files = sorted(glob.glob(pattern))
        if frame_files:
            print(f"Found frames using pattern: {pattern}")
            break
    
    if not frame_files:
        print(f"No frames found in {input_dir}")
        print("Tried patterns:")
        for pattern in frame_patterns:
            print(f"  - {pattern}")
        return False
    
    print(f"Found {len(frame_files)} frames")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Write frames to video
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        video_writer.write(frame)
        
        if i % 50 == 0:
            print(f"Processed {i}/{len(frame_files)} frames")
    
    # Release everything
    video_writer.release()
    # Removed cv2.destroyAllWindows() to avoid headless environment errors
    
    print(f"Video saved to {output_video}")
    return True


def frames_to_video_ffmpeg(input_dir, output_video, fps=30):
    """Convert frames to video using system ffmpeg via subprocess (most robust)."""
    frame_patterns = [
        os.path.join(input_dir, "rgb", "rgb_*.png"),
        os.path.join(input_dir, "rgb_*.png"),
        os.path.join(input_dir, "*.png"),
    ]

    frame_files = []
    for pattern in frame_patterns:
        frame_files = sorted(glob.glob(pattern))
        if frame_files:
            print(f"Found frames using pattern: {pattern}")
            break

    if not frame_files:
        print(f"No frames found in {input_dir}")
        print("Tried patterns:")
        for pattern in frame_patterns:
            print(f"  - {pattern}")
        return False

    print(f"Found {len(frame_files)} frames")

    # Write a concat list to preserve exact ordering
    concat_list_path = os.path.join(input_dir, "frames_list.txt")
    try:
        with open(concat_list_path, "w") as f:
            for p in frame_files:
                f.write(f"file '{p}'\n")
        cmd = [
            "ffmpeg", "-y",
            "-r", str(fps),
            "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_video,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("ffmpeg not found on PATH. Please install ffmpeg or choose another method.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed with error code {e.returncode}")
        return False
    finally:
        # Keep the list file for debugging by default; uncomment to remove
        # try:
        #     os.remove(concat_list_path)
        # except OSError:
        #     pass
        pass

    print(f"Video saved to {output_video}")
    return True


def frames_to_video_imageio(input_dir, output_video, fps=30):
    """Convert frames to video using imageio"""
    # Try different frame patterns to handle various directory structures
    frame_patterns = [
        os.path.join(input_dir, "rgb", "rgb_*.png"),  # Standard Isaac Sim structure
        os.path.join(input_dir, "rgb_*.png"),         # Direct directory structure
        os.path.join(input_dir, "*.png"),             # Any PNG files
    ]
    
    frame_files = []
    for pattern in frame_patterns:
        frame_files = sorted(glob.glob(pattern))
        if frame_files:
            print(f"Found frames using pattern: {pattern}")
            break
    
    if not frame_files:
        print(f"No frames found in {input_dir}")
        print("Tried patterns:")
        for pattern in frame_patterns:
            print(f"  - {pattern}")
        return False
    
    print(f"Found {len(frame_files)} frames")
    
    try:
        # Try using imageio v2 for compatibility
        import imageio.v2 as imageio_v2
        
        # Force FFMPEG backend by using the URI scheme to avoid falling back to TIFF writer
        ffmpeg_uri = f"ffmpeg:{output_video}"
        with imageio_v2.get_writer(ffmpeg_uri, fps=fps, codec='libx264', quality=8) as writer:
            for i, frame_file in enumerate(frame_files):
                frame = imageio_v2.imread(frame_file)
                writer.append_data(frame)  # type: ignore[attr-defined]
                
                if i % 50 == 0:
                    print(f"Processed {i}/{len(frame_files)} frames")
        
        print(f"Video saved to {output_video}")
        return True
        
    except Exception as e:
        print(f"ImageIO with FFMPEG failed: {e}")
        print("Falling back to system ffmpeg...")
        ffmpeg_ok = frames_to_video_ffmpeg(input_dir, output_video, fps)
        if ffmpeg_ok:
            return True
        print("Falling back to OpenCV method...")
        return frames_to_video_opencv(input_dir, output_video, fps)


def main():
    parser = argparse.ArgumentParser(description="Convert recorded frames to video")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing recorded frames")
    parser.add_argument("--output", type=str, default="",
                       help="Output video filename")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second for output video")
    parser.add_argument("--method", choices=['opencv', 'imageio', 'ffmpeg', 'auto'], default='auto',
                       help="Method to use for video conversion (auto=imageio->ffmpeg->opencv)")
    
    args = parser.parse_args()
    if args.output == "":
        args.output = f"{args.input_dir}/run.mp4"
        print(f"Saving video to input directory: {args.output}")
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Input directory {args.input_dir} does not exist")
        return
    
    # Convert frames to video
    if args.method == 'opencv':
        success = frames_to_video_opencv(args.input_dir, args.output, args.fps)
    elif args.method == 'imageio':
        success = frames_to_video_imageio(args.input_dir, args.output, args.fps)
    elif args.method == 'ffmpeg':
        success = frames_to_video_ffmpeg(args.input_dir, args.output, args.fps)
    else:  # auto
        print("Using auto mode: trying imageio (FFMPEG), then system ffmpeg, then opencv...")
        success = frames_to_video_imageio(args.input_dir, args.output, args.fps)
    
    if success:
        print("Video conversion completed successfully!")
    else:
        print("Video conversion failed!")


if __name__ == "__main__":
    main()
    
    # (env_isaacsim) [evrond@cs-4090-07 rl_for_curobo]$ python projects_root/experiments/core_api/motion_planner/convert_frames_to_video.py --input_dir /cs_storage/evrond/_out_sdrec
    # python projects_root/experiments/core_api/motion_planner/convert_frames_to_video.py --input_dir /cs_storage/evrond/_out_sdrec --method auto
    # search output under rl_for_curobo/