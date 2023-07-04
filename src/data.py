from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple
import h5py as h5p
import numpy as np
import pandas as pd
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
    def __init__(self,session_type: str, session_ID: str, video_folder: Path, annot_path: Path, parts_df: pd.DataFrame,sess_df: pd.DataFrame) -> None:

        self.session_ID = session_ID

        assert session_type in ["TALK","LEGO","GHOST","ANIMALS"]
        self.sess_type = session_type

        self.annot_path = annot_path
        self.annot_hdf = h5p.File('train/annotations/animals_annotations_train/002003/FC1_A/annotations_raw.hdf5')


        self.audio_sample_rate = 44100
        self.video_sample_rate = 25
        self.video_batch_size = 64 # with a stride of 2, this will be approx 2.5 sec of video from the session
        self.audio_batch_size = 132300 # at 44100 khz, this will be approx 2.5 sec of audio from the session
        self.lc_video = VideoReader(str(video_folder / "FC1_A.mp4"))
        self.lc_audio = AudioReader(str(video_folder / "FC1_A.mp4"),sample_rate=self.audio_sample_rate,mono=False)
        self.ec_video = VideoReader(str(video_folder / "FC2_A.mp4"))
        self.sess_df = sess_df[sess_df['ID'] == session_ID]
        self.part1 = parts_df[parts_df['ID' == ["PART.1"]]]
        self.part2 = parts_df[parts_df['ID' == ["PART.2"]]]
        self.video_len = len(self.lc_video)
        self.audio_len = self.lc_audio.shape[1]


        self.age_ten_p1 = torch.tensor(self.part1['AGE'],dtype=torch.float32)
        self.age_ten_p2 = torch.tensor(self.part2['AGE'],dtype=torch.float32)
        self.gender_ten_p1 = torch.tensor(self.part1['GENDER'],dtype=torch.float32)
        self.gender_ten_p2 = torch.tensor(self.part2['GENDER'],dtype=torch.float32)

        self.cult_background_ten_p1 = torch.zeros(6,dtype=torch.float32) #TODO to be done!
        self.cult_background_ten_p2 = torch.zeros(6,dtype=torch.float32) #TODO to be done!

        colname = self.part1.columns[self.part1.eq(self.sess_df['ID']).any()]
        sess_num = int(colname[0][-1:])
        self.session_num_ix = torch.tensor((sess_num - 1) / 4, dtype=torch.float32)


        self.pre_sess_mood_cols_p1 = ['PEQPN_GOOD_BEFORE_PART.1',
                                      'PEQPN_BAD_BEFORE_PART.1',
                                      'PEQPN_HAPPY_BEFORE_PART.1',
                                      'PEQPN_SAD_BEFORE_PART.1',
                                      'PEQPN_FRIENDLY_BEFORE_PART.1',
                                      'PEQPN_UNFRIENDLY_BEFORE_PART.1',
                                      'PEQPN_TENSE_BEFORE_PART.1',
                                      'PEQPN_RELAXED_BEFORE_PART.1']
        self.pre_sess_mood_cols_p2 = [col.replace('1','2') for col in self.pre_sess_mood_cols_p1]

        self.pre_sess_mood_ten_p1 = torch.tensor(self.sess_df[self.pre_sess_mood_cols_p1], dtype=torch.float32)
        self.pre_sess_mood_ten_p2 = torch.tensor(self.sess_df[self.pre_sess_mood_cols_p2], dtype=torch.float32)

        self.pre_sess_fatig_ten_p1 = torch.tensor(self.sess_df['FATIGUE_BEFORE_PART.1'],dtype=torch.float32)
        self.pre_sess_fatig_ten_p2 = torch.tensor(self.sess_df['FATIGUE_BEFORE_PART.2'],dtype=torch.float32)


        self.order_column = [f"{self.sess_type}_ORDER"]
        self.order_ten = torch.tensor(self.sess_df[f"{self.sess_type}_ORDER"],dtype=torch.float32)

        # TODO fill df with other task difficulty values with 0, only lego and animal difficulty has nonzero values rn
        self.difficulty_ten = torch.tensor(self.sess_df[self.sess_type]/3,dtype=torch.float32)
        self.relship_ten = torch.tensor(self.sess_df["KNOWN"],dtype=torch.float32)


        """face_landmarks_tmp: List[Tensor] = []

        with h5py.File(str(annot_path)) as f:
            for frame in f.keys():
                face_landmarks_tmp.append(torch.tensor(f[frame]['face']['landmarks']))
        
        self.face_landmarks: Tensor = torch.stack(face_landmarks_tmp)
        """
    def prepare_metadata(self) -> torch.Tensor:
        p1_mtdt_ten = torch.stack([
            self.age_ten_p1,
            self.gender_ten_p1,
            self.cult_background_ten_p1,
            self.session_num_ix,
            self.pre_sess_mood_ten_p1,
            self.pre_sess_fatig_ten_p1,
            self.difficulty_ten,
            self.relship_ten
        ])
        return p1_mtdt_ten
    def __len__(self):
        return self.video_len - 64

    def __getitem__(self, idx) -> Tuple[Tensor,torch.Tensor,np.ndarray]:

        lc_video_ix_list = list(range(idx, idx + self.video_batch_size, 2))
        lc_video_batch = self.lc_video.get_batch(lc_video_ix_list)
        
        ec_video_batch = self.lc_video.get_batch(lc_video_ix_list)

        part_ratio = idx / len(self.lc_video)

        lc_audio_batch_start = int(part_ratio * self.lc_audio.shape[1])
        lc_audio_batch_end = lc_audio_batch_start + self.audio_batch_size

        audio_batch_ix_list = list(range(lc_audio_batch_start,lc_audio_batch_end,1))
        audio_batch = self.lc_audio.get_batch(audio_batch_ix_list)
