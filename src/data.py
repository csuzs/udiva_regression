from pathlib import Path
from typing import Any, List
import torch
from torch.utils.data import Dataset
import decord
from decord import VideoReader
from decord import cpu, gpu
import h5py
from torch import Tensor
decord.bridge.set_bridge('torch')
        
class Session:
    def __init__(self,session_ID: str, video_path: Path, annot_path: Path, metadata_path: Path ) -> None:
        self.vr = VideoReader(str(video_path))
        self.annot_path = annot_path
        self.mtdt_path = metadata_path

        face_landmarks_tmp: List[Tensor] = []
        
        with h5py.File(str(annot_path)) as f:
            for frame in f.keys():
                face_landmarks_tmp.append(torch.tensor(f[frame]['face']['landmarks']))
        
        self.face_landmarks: Tensor = torch.stack(face_landmarks_tmp)
        
    
class UDIVADataset(Dataset):
    def __init__(self,database_path: Path):
        self.
    
    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)