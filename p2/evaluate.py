import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm.auto import tqdm


def config_parser():
    """Define command line arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--image_path", required=True, help="config file path")
    parser.add_argument("--csv_path", required=True, help="config file path")
    parser.add_argument(
        "--output_path", default="./out.csv", required=True, help="config file path"
    )
    parser.add_argument("--model_path1", required=True, help="config file path")
    parser.add_argument("--model_path2", required=True, help="config file path")
    # parser.add_argument("--model_path3", required=True, help="config file path")

    return parser


test_tfm = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


train_tfm = transforms.Compose(
    [
        # Resize the image into a fixed shape (height = width = 128)
        transforms.Resize((128, 128)),
        # You may add some transforms here.
        # ToTensor() should be the last one of the transforms.
        transforms.RandomApply(
            transforms=[
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=(128, 128)),
            ],
            p=0.8,
        ),
        transforms.RandomApply(
            transforms=[
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.5, saturation=0.2, hue=0.1
                ),
                transforms.RandomEqualize(),
                transforms.RandomSolarize(threshold=100.0),
            ],
            p=0.4,
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


myseed = 1314520  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
device = "cuda" if torch.cuda.is_available() else "cpu"
parser = config_parser()
args = parser.parse_args()


with open("./p2/class.json", newline="") as jsonfile:
    classes = json.load(jsonfile)


class FinetuneDataset:
    def __init__(self, imgpath, csv, tfm=test_tfm):
        # self.files =

        sort_csv = pd.read_csv(csv)
        self.labels_list = sorted(sort_csv.label.values[:])

        self.filenames = sorted(sort_csv.filename.values[:])
        self.images_list = [
            os.path.join(imgpath, x)
            for x in sorted(os.listdir(imgpath))
            if x in self.filenames
        ]
        self.transform = tfm

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        im = self.transform(Image.open(self.images_list[idx]))
        # source train and valid / target valid -> image and label
        # stripped_label = self.images_list[idx].split("/")[-1][:-9]
        # label = classes[stripped_label]

        # target train -> only image
        # print(self.images_list[idx], stripped_label, label)
        return im, self.filenames[idx]
        # return im, label


valid_set = FinetuneDataset(args.image_path, args.csv_path, tfm=test_tfm)
valid_loader = DataLoader(
    valid_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)


class fullModel(nn.Module):
    def __init__(self) -> None:
        super(fullModel, self).__init__()
        self.backbone = models.resnet50(pretrained=False)

        self.fc = nn.Sequential(
            nn.ReLU(), nn.Linear(1000, 250), nn.ReLU(), nn.Linear(250, 65)
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.fc(out)
        return out


class EnsembledModel(nn.Module):
    def __init__(self, model1, model2) -> None:
        super(EnsembledModel, self).__init__()
        self.model1 = fullModel()
        self.model1.load_state_dict(
            torch.load(model1, map_location="cpu")["model_state_dict"]
        )
        self.model2 = fullModel()
        self.model2.load_state_dict(
            torch.load(model2, map_location="cpu")["model_state_dict"]
        )
        # self.model3 = fullModel()
        # self.model3.load_state_dict(torch.load(model2)["model_state_dict"])

    def forward(self, x):
        out = self.model1(x) + self.model2(x) * 0.5
        return out


model = EnsembledModel(args.model_path1, args.model_path2).to(device)

template = pd.read_csv(args.csv_path)

# The number of training epochs and patience.
patience = 100  # If no improvement in 'patience' epochs, early stop

# Initialize a model, and put it on the device specified.
# model = models.vgg16_bn(pretrained=False).to(device)
# For the classification task, we use cross-entropy as the measurement of performance.

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0

model.eval()

# These are used to record information in validation.
valid_loss = []
valid_accs = []

template.set_index("filename", inplace=True)
# Iterate the validation set by batches.
for idx, (img, filename) in enumerate(valid_loader):

    # A batch consists of image data and corresponding labels.
    # imgs = imgs.half()

    # We don't need gradient in validation.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits = model(img.to(device))

    # We can still compute the loss (but not the gradient).
    # Compute the accuracy for current batch.
    # acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    label = list(classes.keys())[logits.argmax(dim=-1)]
    template["label"].loc[filename] = label
    # Record the loss and accuracy.
    # valid_accs.append(acc)
    # break

# The average loss and accuracy for entire validation set is the average of the recorded values.
# valid_acc = sum(valid_accs) / len(valid_accs)

#######################
#      scheduler      #
#######################

#######################
# Print the information.
# print(
# f"[ Valid | acc = {valid_acc:.5f}"
# )

template.to_csv(args.output_path)

#
