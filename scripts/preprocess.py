import os
import argparse
import subprocess
import torch
import random
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from typing import Tuple, List, Union
import decord
import json
import cv2
import sys

def convert_video(org_path: str, dst_path: str, vid_list: List[str]) -> None:
    for idx, vid in enumerate(vid_list):
        if vid.endswith('.mp4') or vid.endswith('.flv'):
            org_vid_path = os.path.join(org_path, vid)
            dst_vid_path = os.path.join(dst_path, os.path.splitext(vid)[0]+'.mp4')

            if org_vid_path != dst_vid_path:
                cmd = [
                    "ffmpeg", "-hide_banner", "-y", "-i", org_vid_path,
                    "-r", "25", "-crf", "15", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", dst_vid_path
                ]
                subprocess.run(cmd, check=True)

            if idx % 1000 == 0:
                print(f"### {idx} videos converted ###")
                
def segment_video(org_path: str, dst_path: str, vid_list: List[str], segment_duration: int = 30) -> None:
    for idx, vid in enumerate(vid_list):
        if vid.endswith('.mp4') or vid.endswith('.flv'):
            input_file = os.path.join(org_path, os.path.splitext(vid)[0]+'.mp4')
            original_filename = os.path.basename(input_file)

            command = [
                'ffmpeg', '-i', input_file, '-c', 'copy', '-map', '0',
                '-segment_time', str(segment_duration), '-f', 'segment',
                '-reset_timestamps', '1',
                os.path.join(dst_path, f'clip%03d_{original_filename}')
            ]

            subprocess.run(command, check=True)

def extract_audio(org_path: str, dst_path: str, vid_list: List[str]) -> None:
    for idx, vid in enumerate(vid_list):
        if vid.endswith('.mp4'):
            video_path = os.path.join(org_path, vid)
            audio_output_path = os.path.join(dst_path, os.path.splitext(vid)[0] + ".wav")
            try:
                command = [
                    'ffmpeg', '-hide_banner', '-y', '-i', video_path,
                    '-vn', '-acodec', 'pcm_s16le', '-f', 'wav',
                    '-ar', '16000', '-ac', '1', audio_output_path,
                ]

                subprocess.run(command, check=True)
                print(f"Audio saved to: {audio_output_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error extracting audio from {vid}: {e}")

def generate_train_list(cfg):
    train_file_path = cfg.video_file_list_train
    val_file_path = cfg.video_file_list_val
    test_file_path = cfg.video_file_list_test
    fps_file_path = cfg.video_root_25fps
    fps_file_list = os.listdir(fps_file_path)

    test_ids = ['1005', '1010', '1015', '1020', '1021',
                '1030', '1033', '1046', '1052', '1062',
                '1074', '1081', '1082', '1089']
    train_ids = []
    val_ids = []
    num = 0
    random.seed(42)
    random.shuffle(fps_file_list)
    for f in fps_file_list:
        if f.endswith('mp4'):
            identity, text, emotion, intensity = os.path.basename(f).split('_')
            if identity not in test_ids:
                if num % 9 != 0 and identity not in train_ids:
                    train_ids.append(identity)
                    num = num + 1
                elif num % 9 == 0 and identity not in val_ids:
                    val_ids.append(identity)
                    num = num + 1
                else:
                    continue
            else:
                continue

    train_file_list = []
    val_file_list = []
    test_file_list = []
    for f in fps_file_list:
        if f.endswith('mp4'):
            identity, text, emotion, intensity = os.path.splitext(os.path.basename(f))[0].split('_')
            if intensity == 'HI' or intensity == 'XX':
                if identity in train_ids:
                    train_file_list.append(os.path.basename(f))
                elif identity in val_ids:
                    val_file_list.append(os.path.basename(f))
                else:
                    test_file_list.append(os.path.basename(f))
            else:
                continue
        else:
            continue

    train_file_list = sorted(train_file_list)
    val_file_list = sorted(val_file_list)
    test_file_list = sorted(test_file_list)

    save_list_to_file(train_file_path, train_file_list)
    save_list_to_file(val_file_path, val_file_list)
    save_list_to_file(test_file_path, test_file_list)

    print(f'train num is {len(train_ids)}, val num is {len(val_ids)}, test num is {len(test_ids)}')
    print(val_ids)

def save_list_to_file(file_path: str, data_list: List[str]) -> None:
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(f"{item}\n")
                
def main(cfg):
    # Ensure all necessary directories exist
    os.makedirs(cfg.video_root_25fps, exist_ok=True)
    # os.makedirs(cfg.video_audio_clip_root, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_file_list), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_file_list_train), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_file_list_val), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_file_list_test), exist_ok=True)

    vid_list = os.listdir(cfg.video_root_raw)
    sorted_vid_list = sorted(vid_list)

    # Save video file list
    with open(cfg.video_file_list, 'w') as file:
        for vid in sorted_vid_list:
            file.write(vid + '\n')

    convert_video(cfg.video_root_raw, cfg.video_root_25fps, sorted_vid_list)
    segment_video(cfg.video_root_25fps, cfg.video_audio_clip_root, vid_list, segment_duration=cfg.clip_len_second)

    fps_vid_list = os.listdir(cfg.video_root_25fps)
    extract_audio(cfg.video_root_25fps, cfg.video_root_25fps, fps_vid_list)

    generate_train_list(cfg)
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/preprocess.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)

