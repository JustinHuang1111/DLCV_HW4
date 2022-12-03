import torch
from byol_pytorch import BYOL
from torchvision import models
from tqdm.auto import tqdm

resnet = models.resnet50(pretrained=False)

learner = BYOL(
    resnet,
    image_size=128,
    hidden_layer="avgpool",
    use_momentum=False,  # turn off momentum in the target encoder
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)


def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)


for _ in tqdm(range(100)):
    images = sample_unlabelled_images()
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()

# save your improved network
torch.save(resnet.state_dict(), "./ckpt/improved-net.pt")
