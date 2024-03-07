import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, img_dir):
        classes_ = os.listdir(img_dir)
        self.images = []
        self.classes = []

        for c in classes_:
            self.images += [os.path.join(img_dir, c, i) for i in os.listdir(os.path.join(img_dir, c))]
            self.classes += [int(c) for _ in os.listdir(os.path.join(img_dir, c))]
        self.class_map = {0: "green beans", 1: "corn", 2: "diced carrot", 3: "sliced carrot",
                          5: "broccoli", 6: "cauliflower", 7: "wax beans", 8: "whole baby carrots",
                          9: "lima beans", 10: "romano beans", 11: "red peppers", 12: "onions",
                          13: "zucchini", 14: "peas", 15: "green zucchini", 16: "cut wax beans",
                          17: "whole green beans", 18: "blackfungus", 19: "Horsenettle", 20: "snail",
                          21: "round sliced carrots"}
        self.img_dim = (64, 64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, class_name = self.images[idx], self.classes[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = int(class_name)  # self.class_map[class_name]
        img_tensor = torch.from_numpy((img/255).astype(np.float32))
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor(class_id)
        return img_tensor, class_id


cls_dataset = CustomDataset(img_dir=r"D:\anshul\notera\data\class")
cls_data_loader = DataLoader(cls_dataset, batch_size=16, shuffle=True)
