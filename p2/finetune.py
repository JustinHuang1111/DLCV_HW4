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
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--image_path", required=True, help="config file path")
    parser.add_argument("--model_path", required=True, help="config file path")
    parser.add_argument("--exp_name", required=True, help="config file path")
    parser.add_argument(
        "--n_epochs", default=100, type=int, required=True, help="config file path"
    )

    return parser


test_tfm = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose(
    [
        # Resize the image into a fixed shape (height = width = 128)
        transforms.Resize((128, 128)),
        # You may add some transforms here.
        # ToTensor() should be the last one of the transforms.
        transforms.RandomApply(
            transforms=[
                transforms.RandomHorizontalFlip(),
                # transforms.RandomResizedCrop(size=(128, 128)),
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


with open("./class.json", newline="") as jsonfile:
    classes = json.load(jsonfile)


class FinetuneDataset:
    def __init__(self, datapath, train: bool, tfm=test_tfm):
        # self.files =

        self.is_train = train
        if self.is_train:
            sort_csv = pd.read_csv(os.path.join(datapath, "train.csv")).sort_values(
                "filename"
            )
        else:
            sort_csv = pd.read_csv(os.path.join(datapath, "val.csv")).sort_values(
                "filename"
            )
        self.labels_list = sort_csv.label.values[:]
        print(
            f"finish building label list at {datapath} Training:{self.is_train}, the length is {len(self.labels_list)}"
        )
        self.filenames = sort_csv.filename.values[:]
        print(self.filenames, self.labels_list)
        if self.is_train:
            self.images_list = [
                os.path.join(datapath, "train", x)
                for x in os.listdir(os.path.join(datapath, "train"))
                if x in self.filenames
            ]

        else:
            self.images_list = [
                os.path.join(datapath, "val", x)
                for x in os.listdir(os.path.join(datapath, "val"))
                if x in self.filenames
            ]

        print(
            f"finish building image list at {datapath}, number of images:{len(self.images_list)}"
        )
        self.transform = tfm

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        im = self.transform(Image.open(self.images_list[idx]))

        # source train and valid / target valid -> image and label
        label = classes[self.labels_list[idx]]

        # target train -> only image
        print(self.images_list[idx], self.labels_list[idx], label)
        return im, label


train_set = FinetuneDataset(args.image_path, train=True, tfm=train_tfm)
train_loader = DataLoader(
    train_set, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
)
valid_set = FinetuneDataset(args.image_path, train=False, tfm=test_tfm)
valid_loader = DataLoader(
    valid_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
)


class fullModel(nn.Module):
    def __init__(self) -> None:
        super(fullModel, self).__init__()
        if not args.pretrain:
            self.backbone = models.resnet50(pretrained=False)
            self.backbone.load_state_dict(torch.load(args.model_path))
            print(f"use model from {args.model_path}")
        else:
            self.backbone = models.resnet50(pretrained=True)
            print("use pretrained model")
        print(self.backbone)
        self.fc = nn.Sequential(
            nn.ReLU(), nn.Linear(1000, 250), nn.ReLU(), nn.Linear(250, 65)
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.fc(out)
        return out


model = fullModel().to(device)


# The number of training epochs and patience.
patience = 30  # If no improvement in 'patience' epochs, early stop

# Initialize a model, and put it on the device specified.
# model = models.vgg16_bn(pretrained=False).to(device)
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0
for epoch in range(args.n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))
        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))
        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()
        # Compute the gradients for parameters.
        loss.backward()
        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        # Update the parameters with computed gradients.
        optimizer.step()
        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(
        f"[ Train | {epoch + 1:03d}/{args.n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}"
    )

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        # imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        # break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    #######################
    #      scheduler      #
    #######################
    scheduler.step(valid_loss)

    #######################
    # Print the information.
    print(
        f"[ Valid | {epoch + 1:03d}/{args.n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}"
    )

    # update logs
    if valid_acc > best_acc:
        with open(f"./{args.exp_name}_log.txt", "a") as f:
            print(
                f"[ Valid | {epoch + 1:03d}/{args.n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best"
            )
            f.write(
                f"[ Valid {args.exp_name} | {epoch + 1:03d}/{args.n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best\n"
            )
    else:
        with open(f"./{args.exp_name}_log.txt", "a") as f:
            print(
                f"[ Valid | {epoch + 1:03d}/{args.n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}"
            )
            f.write(
                f"[ Valid {args.exp_name} | {epoch + 1:03d}/{args.n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}\n"
            )

    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            f"./ckpt/{args.exp_name}_finetune.pt",
        )
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break
