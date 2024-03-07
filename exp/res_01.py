import sys
import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append(r"C:\Users\transponster\Documents\anshul\postProcessing")
from model.resnet import ResidualClassifier
from loader.torch import cls_data_loader


model = ResidualClassifier(num_classes=22)
dev = torch.device("cuda")
model.to(dev)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


num_epochs = 100
for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(cls_data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(torch.device("cuda"))
        labels = labels.to(torch.device("cuda"))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print(f'[{epoch + 1}] loss: {running_loss / 2000:.3f}')

print('Finished Training')
model.to(torch.device('cpu'))
torch.save(model.state_dict(), r"D:\anshul\notera\model\post_processing\classifier.pth")
