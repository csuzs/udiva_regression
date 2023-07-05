from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import decord
import h5py as h5p
import numpy as np
import pandas as pd
import torch
from decord import AudioReader, AVReader, VideoReader, cpu, gpu
from torch import Tensor
from torchvision.transforms.functional import resized_crop

decord.bridge.set_bridge("torch")
from torch.utils.data import Dataset


@dataclass
class Session(Dataset):
    def __init__(
        self,
        session_type: str,
        session_ID: str,
        il_rec_path: Path,
        other_il_rec_path: Path,
        il_annot_path: Path,
        other_il_annot_path: Path,
        parts_df: pd.DataFrame,
        sess_df: pd.DataFrame,
    ) -> None:

        self.session_ID = int(session_ID)

        assert session_type in ["TALK", "LEGO", "GHOST", "ANIMALS"]
        self.sess_type = session_type

        self.il_rec_path = il_rec_path
        self.other_il_rec_path = other_il_rec_path
        self.il_annot_path = il_annot_path
        self.other_il_annot_path = other_il_annot_path

        self.il_annot_file = h5p.File(il_annot_path)

        self.audio_sample_rate = 44100
        self.video_sample_rate = 25
        self.video_batch_size = 64  # with a stride of 2, this will be approx 2.5 sec of video from the session
        self.audio_batch_size = 132300  # at 44100 khz, this will be approx 2.5 sec of audio from the session

        self.il_av = AVReader(
            str(il_rec_path), sample_rate=self.audio_sample_rate, mono=False
        )
        self.other_il_av = AVReader(
            str(other_il_rec_path), sample_rate=self.audio_sample_rate, mono=False
        )

        self.sess_df = sess_df[sess_df["ID"] == self.session_ID]
        self.part1 = parts_df[parts_df["ID"] == self.sess_df.iloc[0]["PART.1"]]
        self.part2 = parts_df[parts_df["ID"] == self.sess_df.iloc[0]["PART.2"]]
        self.video_len = len(self.il_av)

        self.age_ten_p1 = torch.tensor([self.part1.iloc[0]["AGE"]], dtype=torch.float32)
        self.age_ten_p2 = torch.tensor([self.part2.iloc[0]["AGE"]], dtype=torch.float32)
        self.gender_ten_p1 = torch.tensor(
            [self.part1.iloc[0]["GENDER"]], dtype=torch.float32
        )
        self.gender_ten_p2 = torch.tensor(
            [self.part2.iloc[0]["GENDER"]], dtype=torch.float32
        )

        self.cult_background_ten_p1 = torch.zeros(
            6, dtype=torch.float32
        )  # TODO to be done!
        self.cult_background_ten_p2 = torch.zeros(
            6, dtype=torch.float32
        )  # TODO to be done!

        colname = self.part1.columns[self.part1.eq(self.session_ID).any()]
        sess_num = int(colname[0][-1:])
        self.session_num_ix = torch.tensor([(sess_num - 1) / 4], dtype=torch.float32)

        self.pre_sess_mood_cols_p1 = [
            "PEQPN_GOOD_BEFORE_PART.1",
            "PEQPN_BAD_BEFORE_PART.1",
            "PEQPN_HAPPY_BEFORE_PART.1",
            "PEQPN_SAD_BEFORE_PART.1",
            "PEQPN_FRIENDLY_BEFORE_PART.1",
            "PEQPN_UNFRIENDLY_BEFORE_PART.1",
            "PEQPN_TENSE_BEFORE_PART.1",
            "PEQPN_RELAXED_BEFORE_PART.1",
        ]
        self.pre_sess_mood_cols_p2 = [
            col.replace("1", "2") for col in self.pre_sess_mood_cols_p1
        ]

        self.pre_sess_mood_ten_p1 = torch.tensor(
            self.sess_df.iloc[0][self.pre_sess_mood_cols_p1], dtype=torch.float32
        )
        self.pre_sess_mood_ten_p2 = torch.tensor(
            self.sess_df.iloc[0][self.pre_sess_mood_cols_p2], dtype=torch.float32
        )

        self.pre_sess_fatig_ten_p1 = torch.tensor(
            [self.sess_df.iloc[0]["FATIGUE_BEFORE_PART.1"]], dtype=torch.float32
        )
        self.pre_sess_fatig_ten_p2 = torch.tensor(
            [self.sess_df.iloc[0]["FATIGUE_BEFORE_PART.2"]], dtype=torch.float32
        )

        self.order_column = [f"{self.sess_type}_ORDER"]
        self.order_ten = torch.tensor(
            [self.sess_df.iloc[0][f"{self.sess_type}_ORDER"]], dtype=torch.float32
        )
        ocean_cols = [
            "OPENMINDEDNESS_Z",
            "CONSCIENTIOUSNESS_Z",
            "EXTRAVERSION_Z",
            "AGREEABLENESS_Z",
            "NEGATIVEEMOTIONALITY_Z",
        ]
        self.part1_ocean_score = torch.tensor(
            self.part1.iloc[0][ocean_cols], dtype=torch.float32
        )

        # TODO fill df with other task difficulty values with 0, only lego and animal difficulty has nonzero values rn
        self.difficulty_ten = torch.tensor(
            [self.sess_df.iloc[0][self.sess_type] / 3], dtype=torch.float32
        )
        self.relship_ten = torch.tensor(
            [self.sess_df.iloc[0]["KNOWN"]], dtype=torch.float32
        )

    def prepare_metadata(self) -> Tuple[Tensor, Tensor]:

        metadata_list_p1 = [
            self.age_ten_p1,
            self.gender_ten_p1,
            self.cult_background_ten_p1,
            self.session_num_ix,
            self.pre_sess_mood_ten_p2,
            self.pre_sess_fatig_ten_p2,
            self.difficulty_ten,
            self.relship_ten,
        ]
        metadata_list_p2 = [
            self.age_ten_p2,
            self.gender_ten_p2,
            self.cult_background_ten_p2,
            self.session_num_ix,
            self.pre_sess_mood_ten_p1,
            self.pre_sess_fatig_ten_p1,
            self.difficulty_ten,
        ]

        p1_mtdt_ten = torch.cat(metadata_list_p1)

        p2_mtdt_ten = torch.cat(metadata_list_p2)
        return (p1_mtdt_ten, p2_mtdt_ten)

    def __len__(self):
        return self.video_len - self.video_batch_size

    def __getitem__(
        self, idx
    ) -> Tuple[
        Tuple[Tensor, Tensor, np.ndarray, Tensor, Tensor], Tuple[Tensor, Tensor]
    ]:
        # local context chunk = lcc
        # extended context chunk = ecc

        video_ix_list = list(range(idx, idx + self.video_batch_size, 2))
        lcc_audio, lcc_video = self.il_av.get_batch(video_ix_list)

        face_landmarks = [self.il_annot_file[str(ix).zfill(5)] for ix in video_ix_list]

        # removing depth information and converting to tensor
        face_landmarks_ten = torch.stack(
            [torch.asarray(fl["face"]["landmarks"][:, :2]) for fl in face_landmarks]
        )
        min_vals, max_vals = face_landmarks_ten.min(dim=1), face_landmarks_ten.max(
            dim=1
        )
        diffs = max_vals.values - min_vals.values

        # N,W,H,C, resizedCrop doesn't work on batches of images
        faceimg_batch = torch.stack(
            [
                resized_crop(
                    img, minv[1], minv[0], d[1], d[0], list(lcc_video.shape)[-2:]
                )
                for img, minv, d in zip(lcc_video, min_vals.values, diffs)
            ]
        ).permute([0, 2, 3, 1])

        _, ecc_video = self.other_il_av.get_batch(video_ix_list)
        lcc_mtdt, ecc_mtdt = self.prepare_metadata()

        return (
            (lcc_video, faceimg_batch, lcc_video, lcc_mtdt, self.part1_ocean_score),
            (ecc_video, ecc_mtdt),
        )


def preprocess_dataframes(sessions: pd.DataFrame, parts: pd.DataFrame):

    parts["AGE"] = (parts["AGE"] - parts["AGE"].min()) / (
        parts["AGE"].max() - parts["AGE"].min()
    )
    parts["GENDER"] = parts["GENDER"].map({"F": 0, "M": 1})
    pre_sess_mood_cols_p1 = [
        "PEQPN_GOOD_BEFORE_PART.1",
        "PEQPN_BAD_BEFORE_PART.1",
        "PEQPN_HAPPY_BEFORE_PART.1",
        "PEQPN_SAD_BEFORE_PART.1",
        "PEQPN_FRIENDLY_BEFORE_PART.1",
        "PEQPN_UNFRIENDLY_BEFORE_PART.1",
        "PEQPN_TENSE_BEFORE_PART.1",
        "PEQPN_RELAXED_BEFORE_PART.1",
    ]
    pre_sess_mood_cols_p2 = [col.replace("1", "2") for col in pre_sess_mood_cols_p1]
    sessions[pre_sess_mood_cols_p1 + pre_sess_mood_cols_p2] = (
        sessions[pre_sess_mood_cols_p1 + pre_sess_mood_cols_p2] - 1
    ) / 4
    sessions["FATIGUE_BEFORE_PART.1"] = sessions["FATIGUE_BEFORE_PART.1"] / 10
    sessions["FATIGUE_BEFORE_PART.2"] = sessions["FATIGUE_BEFORE_PART.2"] / 10
    order_cols = [
        "TALK_ORDER",
        "LEGO_ORDER",
        "GHOST_ORDER",
        "GAZE_ORDER",
        "ANIMALS_ORDER",
    ]
    sessions[order_cols] = (sessions[order_cols] - 1) / 3
    difficulty_cols = ["LEGO", "ANIMALS"]
    sessions[difficulty_cols] = sessions[difficulty_cols] / 3
    sessions["KNOWN"] = sessions["KNOWN"].astype(int)


def get_train_datasets(
    session_task: str,
    annotations_path: Path,
    recordings_path: Path,
    metadata_path: Path,
) -> List[Dataset]:

    parts_df = pd.read_csv(metadata_path / "parts_train.csv")
    sess_df = pd.read_csv(metadata_path / "sessions_train.csv")

    preprocess_dataframes(sess_df, parts_df)

    annot_files = [
        p
        for p in annotations_path.rglob("*")
        if not p.name.startswith(".") and p.name.endswith("hdf5")
    ]
    parts_names = [p.parts[-2] for p in annot_files]
    sess_ids = [p.parts[-3] for p in annot_files]
    recordings = [
        recordings_path / sess_id / f"{part_name}.mp4"
        for sess_id, part_name in zip(sess_ids, parts_names)
    ]
    assert len(parts_names) == len(sess_ids) == len(recordings)

    sessions: List[Session] = []

    for i in range(0, len(parts_names), 2):
        sess = Session(
            session_type=session_task,
            session_ID=sess_ids[i],
            il_rec_path=recordings[i],
            other_il_rec_path=recordings[i + 1],
            il_annot_path=annot_files[i],
            other_il_annot_path=annot_files[i + 1],
            parts_df=parts_df,
            sess_df=sess_df,
        )
        other_sess = Session(
            session_type=session_task,
            session_ID=sess_ids[i],
            il_rec_path=recordings[i + 1],
            other_il_rec_path=recordings[i],
            il_annot_path=annot_files[i + 1],
            other_il_annot_path=annot_files[i],
            parts_df=parts_df,
            sess_df=sess_df,
        )
        sessions.append(sess)
        sessions.append(other_sess)

    return sessions


"""
if __name__ == "__main__":
    abs_path_prefix = (
        "/Users/zsomborcsurilla/Documents/elte_msc/2023_tavasz/advml/udiva_regression"
    )
    train_datasets = get_train_datasets(
        session_task="ANIMALS",
        annotations_path=abs_path_prefix
        / Path("train/annotations/animals_annotations_train"),
        recordings_path=abs_path_prefix
        / Path("train/recordings/animals_recordings_train"),
        metadata_path=abs_path_prefix / Path("train/metadata/metadata_train"),
    )

    ((il_video_batch, faceimg_batch, il_audio_batch), other_il_video_batch) = train_datasets[0][0]
"""
