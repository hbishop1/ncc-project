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

    def __init__(self, hierachical=True, reversed_weights=False):
        super(Heirachical_Loss,self).__init__()
        with open('heirachy_graph.p', 'rb') as fp:
            self.hierachy_G = pickle.load(fp)

        inv = {}
        for k, v in self.hierachy_G.items():
            inv[v] = inv.get(v, [])
            inv[v].append(k)
        self.inv_hierachy_G = inv

        graph = {i:81 for i in range(81)}  
        graph[81] = None

        self.flat_G = graph

        inv = {}
        for k, v in self.flat_G.items():
            inv[v] = inv.get(v, [])
            inv[v].append(k)
        self.inv_flat_G = inv

        self.hierachy = hierachical

        self.reversed = reversed_weights



    def forward(self,outputs,target):

        loss = 0
        total_dist = 0
        preds = []
        sftmax = F.softmax(outputs,dim=1)

        graph = self.hierachy_G if self.hierachy else self.flat_G
        inv_graph = self.inv_hierachy_G if self.hierachy else self.inv_flat_G

        for i in range(len(target)):
            probs = {x:0 for x in graph.keys()}
            for l, val in enumerate(sftmax[i]):
                node = l
                probs[node] = val
                while graph[node] != None:
                    node = graph[node]
                    probs[node] += val
            
            node = int(target[i])
            path = []
            while graph[node] != None:
                path = [node] + path
                node = graph[node]

            path = [node] + path
            
            if self.reversed:
                win = sum([(2 ** (j-len(path)))*probs[path[j]] for j in range(len(path))])
                win += 2 ** -len(path) * probs[int(target[i])]
                loss += -(torch.log((win-(2**-len(path)))/(1-2**(-len(path)))) / len(target))
                
            else:
                win = sum([(2 ** -(j+1))*probs[path[j]] for j in range(len(path))])
                win += 2 ** -len(path) * probs[int(target[i])]
                loss += -(torch.log(2*(win-0.5)) / len(target))


            pred = inv_graph[None][0]
            while pred in inv_graph.keys():
                pred = max(inv_graph[pred], key=lambda x : probs[x])
            preds.append(pred)


            node1, node2 = pred, int(target[i])
            while node1 != node2:
                total_dist += 1
                node1, node2 = self.hierachy_G[node1], self.hierachy_G[node2]

        return loss, torch.LongTensor(preds), torch.tensor(total_dist)

    def flat_graph(self):
        self.hierachy = False

    def hierachy_graph(self):
        self.hierachy = True



def imshow(inp):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std * inp + mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    plt.show()


def train_model(model, criterion, optimizer, num_epochs=25, outfile='results'):
    since = time.time()
    logs = {'train_acc':[],'train_loss':[],'test_acc':[],'test_loss':[],'train_dist':[],'test_dist':[]}
    best_acc = 0.0

    open(outfile + '.txt','w')

    ce = nn.CrossEntropyLoss()

    for epoch in range(num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # if epoch % 10 == 0 and epoch != 0:
        #     with open(outfile + '.txt','a') as results:
        #         results.write('Switching to flat graph \n')
        #     criterion.flat_graph()
        # elif epoch % 10 == 1:
        #     with open(outfile + '.txt','a') as results:
        #         results.write('Switching to hierachical graph \n')
        #     criterion.hierachy_graph()

        # with open(outfile + '.txt','a') as results:
        #     results.write('Epoch {}/{} \n'.format(epoch,num_epochs))
        

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
                    print(loss)
                    print(ce(outputs,labels))

                    preds = preds.to(device)

                    # backward + optimize only if in training phase
                    if phase == 'train' and epoch != 0:
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

            with open(outfile + '.txt','a') as results:
                results.write('{} Loss: {:.4f} Acc: {:.4f} Dist: {:.4f} \n'.format(
                    phase, epoch_loss, epoch_acc, epoch_dist))

            logs[phase + '_acc'].append(epoch_acc)
            logs[phase + '_loss'].append(epoch_loss)
            logs[phase + '_dist'].append(epoch_dist)

    with open(outfile + '.p','wb') as fp:
        pickle.dump(logs, fp)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


if __name__ == '__main__':

    learning_rate = 1e-5
    training_iterations = 200

    out = 'test'

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

    # ------- feature extraction --------

    # for param in model.parameters():
    #     param.requires_grad = False


    # ------- alexnet ------------

    model = models.alexnet(pretrained=True)

    model.classifier[-1] = nn.Linear(4096, len(class_names))

    # -------- resnet -------------

    # model = models.resnet18(pretrained=True)

    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs,len(class_names))

    # -------- vgg16 -------------

    # model = models.vgg16_bn(pretrained=True)

    # model.classifier[-1] = nn.Linear(4096, len(class_names))

    # --------------------------------

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)

    criterion = Heirachical_Loss(hierachical=False, reversed_weights=True)

    optimizer = optim.Adam(model.parameters(),lr = learning_rate,weight_decay=0.01)

    train_model(model, criterion, optimizer, training_iterations, out)




