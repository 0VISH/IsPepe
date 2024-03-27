import torch
from torchvision import datasets, transforms
from torch import nn, optim
from sys import argv

shouldVal = False
if "val" in argv: shouldVal=True

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
            #nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            #nn.BatchNorm1d(1024),
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
DELTA = 1e-6

trainBatchSize = 16
testBatchSize = 20
valBatchSize = 20

trainDataset = datasets.ImageFolder(root="data/train/", transform=dataTransform)
trainDataloader = torch.utils.data.DataLoader(trainDataset, batch_size=trainBatchSize, shuffle=True)
testDataset = datasets.ImageFolder(root="data/test/", transform=dataTransform)
testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=testBatchSize, shuffle=True)
if(shouldVal):
    valDataset = datasets.ImageFolder(root="data/val/", transform=dataTransform)
    valDataloader = torch.utils.data.DataLoader(valDataset, batch_size=valBatchSize, shuffle=True)

def calcAccPrecRecf1(outputs, labels, batchSize):
    truePositive = (outputs > 0.5) & (labels == 1)
    trueNegative = (outputs < 0.5) & (labels == 0)
    falseNegative = (outputs < 0.5) & (labels == 1)
    falsePositive = (outputs > 0.5) & (labels == 0)

    #adding DELTA to avoid div by 0 error
    prec = torch.sum(truePositive).item()/(torch.sum(truePositive | falsePositive).item() + DELTA)
    rec = torch.sum(truePositive).item()/(torch.sum(truePositive | falseNegative).item() + DELTA)
    acc = torch.sum(truePositive | trueNegative).item()/batchSize
    f1 = (2 * prec * rec)/(prec + rec + DELTA)
    return acc, prec, rec, f1
def testModel(model, dataloader, batchSize, phase):
    with torch.no_grad():
        criterion = nn.BCELoss()
        loss = 0
        acc = 0
        prec = 0
        rec = 0
        f1 = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.reshape((batchSize, 1)).to(torch.float32)

            outputs = model(images)
            loss += criterion(outputs, labels).item()
            a,p,r,f = calcAccPrecRecf1(outputs, labels, batchSize)
            acc += a
            prec += p
            rec += r
            f1 += f
        count = len(dataloader.dataset)/batchSize
        loss = loss/count
        accuracy = acc/count
        print(f"{phase} Loss: {loss}")
        print(f"{phase} Accuracy: {accuracy}")
        print(f"{phase} Precision: {prec/count}")
        print(f"{phase} Recall: {rec/count}")
        print(f"{phase} F1: {f1/count}")
        return loss, accuracy
def trainModel():
    model = Network()
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay = 0.07, momentum = 0.1)

    trainLosses = []
    trainAccuracies = []
    valLosses = []
    valAccuracies = []
    for epoch in range(35):
        acc = 0
        rec = 0
        prec = 0
        f1 = 0
        runningLoss  = 0
        for images, labels in trainDataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            labels = labels.reshape((trainBatchSize, 1)).to(torch.float32)
            loss = criterion(outputs, labels)

            a,p,r,f = calcAccPrecRecf1(outputs, labels, trainBatchSize)
            acc += a
            prec += p
            rec += r
            f1 += f

            print(f"Epoch: {epoch}", loss.item())
            runningLoss += loss.item()
                    
            loss.backward()
            optimizer.step()
        count = len(trainDataloader.dataset)/trainBatchSize
        runningLoss = runningLoss/count
        accuracy = acc/count
        print("Loss:", runningLoss)
        print("Accuracy:", accuracy)
        print("Precision:", prec/count)
        print("Recall:", rec/count)
        print("F1:", f1/count)
        trainLosses.append(runningLoss)
        trainAccuracies.append(accuracy)
        if(shouldVal):
            valAccuracy, valLoss = testModel(model, valDataloader, valBatchSize, "Val")
            valAccuracies.append(valAccuracy)
            valLosses.append(valLoss)
        torch.cuda.empty_cache()

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

model, trLosses, trAccuracies, vLosses, vAccuracies = trainModel()
print("-" * 75)
testModel(model, testDataloader, testBatchSize, "Test")
plotModel(trLosses, trAccuracies, vLosses, vAccuracies)