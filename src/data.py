from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import decord
from decord import VideoReader, AudioReader
from decord import cpu, gpu
import h5py
from torch import Tensor
decord.bridge.set_bridge('torch')
from torch.utils.data import Dataset

@dataclass


class Session(Dataset):
    def __init__(self,session_ID: str, video_folder: Path, annot_path: Path, metadata_path: Path) -> None:



        self.audio_sample_rate = 44100
        self.video_sample_rate = 25
        self.video_batch_size = 64 # with a stride of 2, this will be approx 2.5 sec of video from the session
        self.audio_batch_size = 132300 # at 44100 khz, this will be approx 2.5 sec of audio from the session
        self.lc_video = VideoReader(str(video_folder / "FC1_A.mp4"))
        self.lc_audio = AudioReader(str(video_folder / "FC1_A.mp4"),sample_rate=self.audio_sample_rate,mono=False)
        self.ec_video = VideoReader(str(video_folder / "FC2_A.mp4"))

        self.annot_path = annot_path
        self.mtdt_path = metadata_path
        self.video_len = len(self.lc_video)
        self.audio_len = self.lc_audio.shape[1]
        
        """face_landmarks_tmp: List[Tensor] = []

        with h5py.File(str(annot_path)) as f:
            for frame in f.keys():
                face_landmarks_tmp.append(torch.tensor(f[frame]['face']['landmarks']))
        
        self.face_landmarks: Tensor = torch.stack(face_landmarks_tmp)
        """
        
    def __len__(self):
        return self.video_len - 64

    def __getitem__(self, idx) -> Tuple[torch.Tensor,torch.Tensor,np.ndarray]:

        lc_video_ix_list = list(range(idx, idx + self.video_batch_size, 2))
        lc_video_batch = self.lc_video.get_batch(lc_video_ix_list)
        
        ec_video_batch = self.lc_video.get_batch(lc_video_ix_list)

        part_ratio = idx / len(self.lc_video)

        lc_audio_batch_start = int(part_ratio * self.lc_audio.shape[1])
        lc_audio_batch_end = lc_audio_batch_start + self.audio_batch_size

        audio_batch_ix_list = list(range(lc_audio_batch_start,lc_audio_batch_end,1))
        audio_batch = self.lc_audio.get_batch(audio_batch_ix_list)


class UDIVADataset(Dataset):
    def __init__(self,database_path: Path):
        self.
    
    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)