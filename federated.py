import torch
import torch.nn as nn
import syft as sy
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import logging
from torch.utils.data import Dataset
# import Pysyft to help us to simulate federated leraning
import syft as sy
hook = sy.TorchHook(torch)

# ### Create nodes
first_node = sy.VirtualWorker(hook=hook, id="first")
second_node = sy.VirtualWorker(hook=hook, id="second")
third_node = sy.VirtualWorker(hook=hook, id="third")
fourth_node = sy.VirtualWorker(hook=hook, id="fourth")
fifth_node = sy.VirtualWorker(hook=hook, id="fifth")
### Define args
batch_size = 2048
test_batch_size = 2048
learning_rate = 0.00001
log_interval = 10
epochs = 10

### Use device
device = torch.device("cpu")

### Read data
data_csv = pd.read_csv("dataset.csv")
# label = data["label"]
# data = list(np.array(data.drop(["label"], axis=1)))
# data_loader = []
# for i in range(len(label)):
#     data_loader.append((torch.tensor([[data[i]]]), label[i]))
# print(len(data_loader))
# torch.save(data_loader, "data.pt")

#data_loader = torch.utils.data.TensorDataset(torch.load("data.pt"))
### Datasets
# create a simple CNN net
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1,padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels = 64, kernel_size = 3, stride = 1,padding = 1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=640, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
        )

        self.dropout = nn.Dropout2d(0.25)
    
    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x,1)
        x = x.view(-1, 640)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
class IDSDataset(Dataset):

    def __init__(self, data, transform=None):
        self.IDS = list(data.values)
        self.transform = transform

        label = []
        image = []

        for i in self.IDS:
            label.append(i[-1])
            image.append(i[0:-1])
        
        self.labels = np.asarray(label).astype('long')
        self.images = np.asarray(image).reshape(-1, 1, 10, 1).astype('float32')
    
    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.images)
def train(model, device, train_loader, optimizer, epoch):
    model.train()

    # iterate over federated data
    for batch_idx, (data, target) in enumerate(train_loader):

        # send the model to the remote location 
        model = model.send(data.location)

        # the same torch code that we are use to
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # this loss is a ptr to the tensor loss 
        # at the remote location
        loss = F.nll_loss(output, target)

        # call backward() on the loss ptr,
        # that will send the command to call
        # backward on the actual loss tensor
        # present on the remote machine
        loss.backward()

        optimizer.step()

        # get back the updated model
        model.get()

        if batch_idx % log_interval == 0:

            # a thing to note is the variable loss was
            # also created at remote worker, so we need to
            # explicitly get it back
            loss = loss.get()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, 
                    batch_idx * batch_size, # no of images done
                    len(train_loader) * batch_size, # total images left
                    100. * batch_idx / len(train_loader), 
                    loss.item()
                )
            )
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # add losses together
            test_loss += F.nll_loss(output, target, reduction='sum').item() 

            # get the index of the max probability class
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def get_data(is_train=True):
    data = IDSDataset(
        data_csv,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((21.8182,), (30.2715,))
        ])
    )
    return data
print(get_data(False))
federated_data = sy.FederatedDataLoader(
    get_data().federate((first_node, second_node, third_node, fourth_node, fifth_node)),
    batch_size=batch_size,
    shuffle=True
)
test_data = torch.utils.data.DataLoader(
    get_data(False),
    batch_size=test_batch_size,
    shuffle=True
)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

logging.info("Starting training !!")

for epoch in range(1, epochs + 1):
    train(model, device, federated_data, optimizer, epoch)
    test(model, device, test_data)
    
# thats all we need to do XD

