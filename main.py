from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler

from src.data import get_train_datasets


def collate_fn(batch):
    # audio tensor should get a padding / cut to a fix size
    return batch


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

    train_dset = ConcatDataset(train_datasets)
    sampler = SubsetRandomSampler(range(len(train_dset)))
    batch_size = 2  # Adjust the batch size as needed
    dataloader = DataLoader(
        train_dset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn
    )

    batch = next(iter(dataloader))
