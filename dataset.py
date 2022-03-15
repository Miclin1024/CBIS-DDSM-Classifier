import torch
from pydicom import dcmread
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import pandas as pd
import os
import urllib.request
import cv2



class CBISDDSMDataset(Dataset):
    def __init__(self, train: bool,
                 transform: transforms = None, target_transform: transforms = None,
                 sample_shape=(299, 299), data_dir="./data"):
        """
        Initialize a torch dataset from
        :param train:
        :param data_dir:
        """
        self.data_label_dict: dict[str, int] = {}
        self.key_list: list[str] = []
        self.transform = transform
        self.target_transform = target_transform
        self.sample_shape = sample_shape

        dcm_dir = os.path.join(os.getcwd(), data_dir, 'train' if train else 'test')

        descriptor_filename = f"mass_case_description_{'train' if train else 'test'}_set.csv"
        descriptor_path = os.path.join(os.getcwd(), descriptor_filename)
        if not os.path.exists(descriptor_path):
            print(f"Downloading {descriptor_filename}...")
            url = f"https://wiki.cancerimagingarchive.net/download/attachments/22516629/{descriptor_filename}"
            with urllib.request.urlopen(url) as response, open(descriptor_filename, "wb") as out_file:
                data = response.read()
                out_file.write(data)
                print("Done.")
        else:
            print(f"Loading {descriptor_path}...")

        description_file = pd.read_csv(descriptor_path)

        for index, row in description_file.iterrows():
            image_folder = os.path.join(dcm_dir, row["image file path"].split("/")[0])
            if not os.path.exists:
                continue
            image_path = list(Path(image_folder).rglob("*.dcm"))
            if len(image_path) == 0:
                continue
            image_path = str(image_path[0])
            breast_density = int(row["breast_density"])
            self.data_label_dict[image_path] = breast_density
            self.key_list.append(image_path)

        print(f"{len(self.key_list)} entries loaded in {'training' if train else 'testing'} dataset")

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        filepath = self.key_list[idx]
        assert os.path.exists(filepath), f"{filepath} was loaded but not found, reload dataset and try again"
        data = dcmread(filepath).pixel_array
        data = cv2.resize(data, dsize=self.sample_shape, interpolation=cv2.INTER_CUBIC)
        data = data.astype("float")

        label = self.data_label_dict[filepath]

        data = transforms.ToTensor()(data)
        data = transforms.Normalize(32768, 128)(data).float()

        label = torch.FloatTensor([1, 0] if label > 2 else [0, 1]).float()

        # if self.transform is not None:
        #     data = self.transform(data)
        #     data.to(torch.float)
        # if self.target_transform is not None:
        #     label = self.target_transform(label)
        #     label.to(torch.float)

        return data, label
