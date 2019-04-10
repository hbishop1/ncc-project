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

        inv = {}
        for k, v in self.G.items():
            inv[v] = inv.get(v, [])
            inv[v].append(k)
        self.inv_G = inv

    def forward(self,outputs,target):

        loss = 0
        total_dist = 0
        preds = []
        sftmax = F.softmax(outputs,dim=1)

        for i in range(len(target)):
            probs = {x:0 for x in self.G.keys()}
            for l, val in enumerate(sftmax[i]):
                node = l
                probs[node] = val
                while self.G[node] != None:
                    node = self.G[node]
                    probs[node] += val
            
            node = int(target[i])
            path = []
            while self.G[node] != None:
                path = [node] + path 
                node = self.G[node]
                
            win = sum([(2 ** -(j+1))*probs[path[j]] for j in range(len(path))])
            win += 2 ** -len(path) * probs[int(target[i])]
            loss += -(torch.log(win) / len(target))

            pred = self.inv_G[None][0]
            while pred in self.inv_G.keys():
                pred = max(self.inv_G[pred], key=lambda x : probs[x])
            preds.append(pred)

            node = pred
            while node not in path and self.G[node] != None:
                total_dist += 1
                node = self.G[node]


        return loss, torch.LongTensor(preds), torch.Tensor([total_dist])

    def flat_graph(self):
        graph = {i:81 for i in range(81)}    # cross entropy
        graph[81] = None
        self.G = graph

        inv = {}
        for k, v in self.G.items():
            inv[v] = inv.get(v, [])
            inv[v].append(k)
        self.inv_G = inv

    def heirachy_graph(self):
        with open('heirachy_graph.p', 'rb') as fp:
            self.G = pickle.load(fp)

        inv = {}
        for k, v in self.G.items():
            inv[v] = inv.get(v, [])
            inv[v].append(k)
        self.inv_G = inv



def imshow(inp):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std * inp + mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    plt.show()


def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_acc = 0.0

    open('results_transfer.txt','w')

    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        if epoch % 20 == 10:
            with open('results_transfer.txt','a') as results:
                results.write('Switching to heirachical graph \n')
            criterion.heirachy_graph()
        elif epoch % 20 == 0 and epoch != 0:
            with open('results_transfer.txt','a') as results:
                results.write('Switching to flat graph \n')
            criterion.flat_graph()

        with open('results_transfer.txt','a') as results:
            results.write('Epoch {}/{} \n'.format(epoch,num_epochs))
        

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_distance = 0

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
                    loss, preds, distance = criterion(outputs, labels)
                    preds = preds.to(device)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_distance += distance.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_dist = running_distance / dataset_sizes[phase]

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), './transfer_model.pt')

            print('{} Loss: {:.4f} Acc: {:.4f} Dist: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_dist))

            with open('results_transfer.txt','a') as results:
                results.write('{} Loss: {:.4f} Acc: {:.4f} Dist: {:.4f} \n'.format(
                    phase, epoch_loss, epoch_acc, epoch_dist))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


if __name__ == '__main__':

    # lr = 5e-6 is best for cross entropy

    learning_rate = 5e-5
    training_iterations = 200

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    data_dir = 'dataset_fine-grained/'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                                shuffle=True)
                for x in ['train', 'test']}
                
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,len(class_names))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_ft = model_ft.to(device)

    #model_ft.load_state_dict(torch.load('./transfer_model_flatgraph.pt',map_location='cpu'))

    criterion = Heirachical_Loss()

    criterion.flat_graph()

    optimizer_ft = optim.Adam(model_ft.parameters(),lr = learning_rate,weight_decay=0.005)

    train_model(model_ft, criterion, optimizer_ft, training_iterations)




