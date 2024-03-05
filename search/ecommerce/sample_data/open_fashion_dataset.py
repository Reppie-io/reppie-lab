from datasets import load_dataset


class OpenFashionDataset:
    def load_dataset(
        self,
        dataset_name: str = "ashraq/fashion-product-images-small",
        split: str = "train",
    ):
        return load_dataset(dataset_name, split=split)
