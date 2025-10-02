# video_utils.py
import numpy as np
from decord import VideoReader, cpu
from PIL import Image
import torch
from image_utils import build_transform, dynamic_preprocess


def get_frame_indices(num_frames, num_segments=8):
    """
    Выбирает индексы кадров равномерно по видео.
    """
    seg_size = float(num_frames) / num_segments
    indices = [int(seg_size * i + seg_size / 2) for i in range(num_segments)]
    indices = [min(idx, num_frames - 1) for idx in indices]
    return indices


def load_video_frames(
    video_file,
    input_size=448,
    max_num=1,
    num_segments=8,
    device="cpu",
    dtype=torch.float32,
):
    """
    Загружает видео и возвращает объединённые тензоры для модели и список количества тайлов на каждый кадр.
    """
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    transform = build_transform(input_size)

    frame_indices = get_frame_indices(num_frames, num_segments)
    pixel_values_list = []
    num_patches_list = []

    for idx in frame_indices:
        img = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        t = torch.stack([transform(tile) for tile in tiles])
        pixel_values_list.append(t)
        num_patches_list.append(t.shape[0])

    pixel_values = torch.cat(pixel_values_list).to(dtype=dtype, device=device)
    return pixel_values, num_patches_list
