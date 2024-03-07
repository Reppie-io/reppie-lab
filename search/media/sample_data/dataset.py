from datasets import load_dataset


class MediaDataset:
    def load_dataset(
        self,
        dataset_name: str = "BEE-spoke-data/medium-articles-en",
        split: str = "train",
    ):
        return load_dataset(dataset_name, split=split)
