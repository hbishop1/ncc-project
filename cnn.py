from __future__ import print_function
import torch
import os
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import pickle
import torchvision
from torchvision import transforms, datasets, models


class Heirachical_Loss(torch.nn.Module):
    def __init__(self):
        super(Heirachical_Loss,self).__init__()
        with open('heirachy_graph.p', 'rb') as fp:
            self.G = pickle.load(fp)

    def forward(self,outputs,target):

        loss = 0
        for i in range(len(target)):
            probs = {x:0 for x in self.G.keys()}
            for l, val in enumerate(outputs[i]):
                node = l
                probs[node] = val
                while self.G[node] != None:
                    node = self.G[node]
                    probs[node] += val
            
            node = int(target[i])
            path = [node]
            while self.G[node] != None:
                node = self.G[node]
                path = [node] + path
                
            win = sum([(2 ** -(j+1))*probs[path[j]] for j in range(len(path))])
            win += 2 ** -len(path) * probs[int(target[i])]
            loss += 1-win

        return loss



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
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        layers.append(nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(1024))

        layers.append(nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm2d(1024))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))


        layers.append(Flatten())

        layers.append(nn.Linear(in_features=1024*8*8, out_features=1024))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(1024))
        layers.append(nn.Dropout(0.5))

        layers.append(nn.Linear(in_features=1024, out_features=num_out))

        self.layers = layers

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_acc = 0.0

    open('results1.txt','w')

    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        with open('results1.txt','a') as results:
            results.write('Epoch {}/{} \n'.format(epoch,num_epochs))
        
        # Each epoch has a training and testing phase
        for phase in ['train', 'test']: 
            if phase == 'train':
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
                    outputs = F.softmax(model(inputs),dim=1)
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

            print('{} Loss: {:.4f} Acc: {:.4f} \n'.format(
                phase, epoch_loss, epoch_acc))

            with open('results1.txt','a') as results:
                results.write('{} Loss: {:.4f} Acc: {:.4f} \n'.format(phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), './model.pt')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


if __name__ == '__main__':

    learning_rate = 0.000005
    training_iterations = 500

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(256),
        torchvision.transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    data_dir = 'dataset_fine-grained/'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                shuffle=True)
                for x in ['train', 'test']}
                
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    model_ft = MyNetwork(len(class_names))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_ft = model_ft.to(device)    

    print('> Size of training dataset: ', len(dataloaders['train'].dataset))
    
    print('> Size of test dataset: ', len(dataloaders['test'].dataset))

    print('> Number of network parameters: ', len(torch.nn.utils.parameters_to_vector(model_ft.parameters())))

    criterion = Heirachical_Loss()

    optimizer_ft = optim.Adam(model_ft.parameters(),lr = learning_rate,weight_decay=0.0)

    train_model(model_ft, criterion, optimizer_ft, training_iterations)



