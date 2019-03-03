import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from matplotlib import pyplot
import numpy

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
nums = list(range(10))

mnistSet = datasets.MNIST(
    "./data/mnist",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]),
)

loader = torch.utils.data.DataLoader(mnistSet, shuffle=True)

# print(len(loader), len(mnistSet[0][0][0][0]))
# for x, y in loader:
#     print(y)


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        # x = x[0]
        # print(len(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        # return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Conv()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

for x, y in loader:
    optimizer.zero_grad()

    out = net(x)
    target = torch.zeros(10)
    target[y] = 1.0
    loss = criterion(out, target)
    print(loss.item())

    loss.backward()
    optimizer.step()

mnistSetEval = datasets.MNIST(
    "./data/mnist", transform=transforms.Compose([transforms.ToTensor()])
)
loader = torch.utils.data.DataLoader(mnistSetEval, shuffle=True)
correct = 0

for x, y in loader:
    with torch.no_grad():
        out = net(x)
        # correct += 1 if out[0][y.item()] > 0.9 else 0
        correct += 1 if nums[numpy.argmax(out[0])] == y.item() else 0

print("accuracy: ", correct / len(mnistSetEval))

