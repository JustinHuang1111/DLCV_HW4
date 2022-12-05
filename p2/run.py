import argparse
import os

import kornia
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from byol_pytorch import BYOL
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
    parser.add_argument(
        "--n_epochs", default=100, type=int, required=True, help="config file path"
    )

    return parser


class TrainDataset(Dataset):
    def __init__(self, path):
        super(Dataset).__init__()
        self.path = path

        self.files = sorted([os.path.join(path, x) for x in os.listdir(path)])
        self.filenames = sorted([file for file in os.listdir(path)])
        print(f"One {path} sample", self.files[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = TF.to_tensor(im)
        return im


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

resnet = models.resnet50(pretrained=False)
os.makedirs("./ckpt", exist_ok=True)
augment_fn = nn.Sequential(kornia.augmentation.RandomHorizontalFlip())

augment_fn2 = nn.Sequential(
    kornia.augmentation.RandomHorizontalFlip(),
    kornia.filters.GaussianBlur2d((3, 3), (1.5, 1.5)),
)

learner = BYOL(
    resnet,
    image_size=128,
    hidden_layer="avgpool",
    # use_momentum=False,  # turn off momentum in the target encoder
    augment_fn=augment_fn,
    augment_fn2=augment_fn2,
).to(device)

optimizer = torch.optim.Adam(learner.parameters(), lr=3e-4)


train_set = TrainDataset(os.path.join(args.image_path))
train_loader = DataLoader(
    train_set, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
)


for epoch in range(args.n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for images in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        # imgs = imgs.half()
        # print(imgs.shape,labels.shape)

        loss = learner(images.to(device))
        # print(np.shape(logits))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.

        # Record the loss and accuracy.
        train_loss.append(loss.item())

    train_loss = sum(train_loss) / len(train_loss)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{args.n_epochs:03d} ] loss = {train_loss:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.

torch.save(resnet.state_dict(), "./ckpt/improved-net.pt")

####################### Finish Pretrain
print("Finished Pretrained")
