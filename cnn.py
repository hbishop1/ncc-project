from __future__ import print_function
import torch
import os
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import torchvision
from torchvision import transforms, datasets, models


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class MyNetwork(nn.Module):
    def __init__(self,num_out):

        super(MyNetwork, self).__init__()
        layers = nn.ModuleList()

        layers.append(nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(128))

        layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        layers.append(nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(384))

        layers.append(nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(384))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        layers.append(nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(512))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        layers.append(nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(1024))    

        layers.append(nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(1024))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        layers.append(Flatten())

        layers.append(nn.Linear(in_features=1024*8*8, out_features=2048))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(2048))

        layers.append(nn.Linear(in_features=2048, out_features=num_out))

        self.layers = layers

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


def visualize_model(model, num_images):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['valid']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def imshow(inp):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std * inp + mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    plt.show()


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_acc = 0.0

    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward(create_graph=False)
                        optimizer.step()

                # statistics
                running_loss += float(loss.item()) * int(inputs.size(0))
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


if __name__ == '__main__':

    learning_rate = 0.002
    training_iterations = 25

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    data_dir = 'dataset_fine-grained/'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'valid']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                shuffle=True)
                for x in ['train', 'valid']}
                
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    model_ft = MyNetwork(len(class_names))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_ft = model_ft.to(device)    

    print('> Number of network parameters: ', len(torch.nn.utils.parameters_to_vector(model_ft.parameters())))

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(),lr = learning_rate,momentum = 0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)


    train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, training_iterations)

    visualize_model(model_ft)



