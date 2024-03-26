import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F

dataTransform = transforms.Compose([
    #data.py resizes all the images
    #transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.BatchNorm1d(256 * 6 * 6),
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

trainBatchSize = 32
testBatchSize = 20
valBatchSize = 20

trainDataset = datasets.ImageFolder(root="data/train/", transform=dataTransform)
trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=trainBatchSize, shuffle=True)
valDataset = datasets.ImageFolder(root="data/val/", transform=dataTransform)
valDataloader = torch.utils.data.DataLoader(valDataset, batch_size=valBatchSize, shuffle=True)
testDataset = datasets.ImageFolder(root="data/test/", transform=dataTransform)
testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=testBatchSize, shuffle=True)

def testModel(model, dataloader, batchSize, phase):
    criterion = nn.BCELoss()
    loss = 0
    correctPreds = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.reshape((batchSize, 1)).to(torch.float32)

        outputs = model(images)
        loss += criterion(outputs, labels).item()
        mask1 = (outputs > 0.5) & (labels == 1)
        mask2 = (outputs < 0.5) & (labels == 0)
        correctPreds += torch.sum(mask1 | mask2).item()
    loss = loss/len(dataloader.dataset)
    accuracy = correctPreds/len(dataloader.dataset)
    print(f"{phase} Loss: {loss}")
    print(f"{phase} Accuracy: {accuracy}")
    return loss, accuracy
def trainModel():
    model = Network()
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay = 0.07, momentum = 0.1)

    runningLoss  = 0
    correctPreds = 0
    trainLosses = []
    trainAccuracies = []
    valLosses = []
    valAccuracies = []
    for epoch in range(35):
        for images, labels in trainDataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            labels = labels.reshape((trainBatchSize, 1)).to(torch.float32)
            loss = criterion(outputs, labels)

            mask1 = (outputs > 0.5) & (labels == 1)
            mask2 = (outputs < 0.5) & (labels == 0)
            correctPreds += torch.sum(mask1 | mask2).item()
            
            print(f"Epoch: {epoch}", loss.item())
            runningLoss += loss.item()
                    
            loss.backward()
            optimizer.step()
        runningLoss = runningLoss/len(trainDataloader.dataset)
        accuracy = correctPreds/len(trainDataloader.dataset)
        print("Loss:", runningLoss)
        print("Accuracy:", accuracy)
        trainLosses.append(runningLoss)
        trainAccuracies.append(accuracy)
        runningLoss = 0
        correctPreds = 0
        valAccuracy, valLoss = testModel(model, valDataloader, valBatchSize, "Val")
        valAccuracies.append(valAccuracy)
        valLosses.append(valLoss)

    torch.save(model.state_dict(), "modelAlex.pth")
    return model, trainLosses, trainAccuracies, valLosses, valAccuracies

def plotModel(trainLosses, trainAccuracies, testLosses, testAccuracies):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2)

    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[1, 0]
    ax4 = ax[1, 1]
    ax1.plot(range(len(trainLosses)), trainLosses, 'r')
    ax2.plot(range(len(trainAccuracies)), trainAccuracies, 'b')
    ax1.title.set_text("loss VS epoch: Train")
    ax2.title.set_text("accuracy VS epoch: Train")
    ax3.plot(range(len(testLosses)), testLosses, 'r')
    ax4.plot(range(len(testAccuracies)), testAccuracies, 'b')
    ax3.title.set_text("loss VS epoch: Validation")
    ax4.title.set_text("accuracy VS epoch: Validation")
    plt.show()

model, trLosses, trAccuracies, teLosses, teAccuracies = trainModel()
testModel(model, testDataloader, testBatchSize, "Test")
plotModel(trLosses, trAccuracies, teLosses, teAccuracies)