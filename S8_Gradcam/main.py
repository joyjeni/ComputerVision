'''Train CIFAR10 with PyTorch.'''
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torchsummary import summary

import matplotlib.pyplot as plt
#%matplotlib inline

import os
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models import resnet
from utils import progress_bar, GradCam, preprocess_image, GuidedBackpropReLUModel, deprocess_image, show_cam_on_image, \
    im_convert
import cv2

import numpy as np

IMAGE_PATH = "./img/misclassified/1.jpg"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

train_transforms = A.Compose([A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                              A.RandomCrop(width=32, height=32, p=1),
                              A.Rotate(limit=5),
                              A.CoarseDropout(max_holes=1, min_holes=1, max_height=16, max_width=16, p=0.5,
                                              fill_value=tuple([x * 255.0 for x in [0.4914, 0.48216, 0.44653]]),
                                              min_height=16, min_width=16),
                              A.Normalize((0.4914, 0.48216, 0.44653), (0.24703, 0.24349, 0.26159)),
                              A.pytorch.ToTensor()
                              ])
# Test Phase transformations
test_transforms = A.Compose([
    #  transforms.Resize((28, 28)),
    #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),

    A.Normalize((0.4914, 0.48216, 0.44653), (0.24703, 0.24349, 0.26159)),
    A.pytorch.ToTensor()
])

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.48216, 0.44653), (0.24703, 0.24349, 0.26159))])
class Transforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=Transforms(transform))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = resnet.ResNet18()
model = model.to(device)

from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []


def train(model, device, train_loader, optimizer, epoch, loss_function):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data['image'].to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = loss_function(y_pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
    train_acc.append(100 * correct / processed)
    train_losses.append(train_loss / len(train_loader))



def test(model, device, test_loader, loss_function):
    model.eval()
    test_loss = 0
    correct = 0
    confusion_matrix = np.zeros([10, 10], int)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

    for i, l in enumerate(target):
        confusion_matrix[l.item(), target[i].item()] += 1




if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):


    train_loss = 0
    correct = 0
    total = 0
    decay = 0
    learning_rate=0.001

    accuracy = 0

    print('\nEpoch: %d' % epoch)
    if (epoch + 1) % 3 == 0:
        decay += 1
        optimizer.param_groups[0]['lr'] = learning_rate * (0.5 ** decay)
        print("The new learning rate is {}".format(optimizer.param_groups[0]['lr']))

    model.train()
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model(inputs)
        batch_loss = criterion(logps, labels)
        batch_loss.backward()
        optimizer.step()

        train_loss += batch_loss.item()
        train_losses.append(train_loss)
        _, predicted = logps.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        accuracy = 100 * correct / total

        progress_bar(batch_idx, len(trainloader), 'batch_loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), accuracy))






def test(epoch):
    global best_acc
    model.eval()
    test_batch_loss = 0.0
    correct = 0.0
    total = 0.0
    accuracy=0.0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            batch_loss = criterion(logps, labels)

            test_batch_loss += batch_loss.item()
            _, predicted = logps.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            accuracy = 100 * correct / total
            test_losses.append(test_batch_loss)
            test_accuracy.append(accuracy)

            progress_bar(batch_idx, len(testloader), 'batch_loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_batch_loss / (batch_idx + 1), accuracy))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 20):
    train_losses = []
    train_accuracy = []
    test_accuracy = []
    test_losses = []
    confusion_matrix = np.zeros([10, 10], int)

    train(epoch)
    test(epoch)
    scheduler.step()
plt.style.use('ggplot')
plt.plot(train_losses, label='training batch_loss')
plt.plot(test_losses, label='validation batch_loss')
plt.legend()

plt.style.use('ggplot')
plt.plot(train_accuracy, label='training accuracy')
plt.plot(test_accuracy, label='validation accuracy')
plt.legend()



summary(model, input_size=(3, 32, 32))




### Visualize

classes = ('airplane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


print('{0:10s} - {1}'.format('Category','Accuracy'))
for i, r in enumerate(confusion_matrix):
    print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
plt.ylabel('Actual Category')
plt.yticks(range(10), classes)
plt.xlabel('Predicted Category')
plt.xticks(range(10), classes)
plt.show()

print('actual/pred'.ljust(16), end='')
for i, c in enumerate(classes):
    print(c.ljust(10), end='')
print()
for i, r in enumerate(confusion_matrix):
    print(classes[i].ljust(16), end='')
    for idx, p in enumerate(r):
        print(str(p).ljust(10), end='')
    print()

    r = r / np.sum(r)
    print(''.ljust(16), end='')
    for idx, p in enumerate(r):
        print(str(p).ljust(10), end='')
    print()



images_batch, labels_batch = iter(trainloader).next()
img = torchvision.utils.make_grid(images_batch)

