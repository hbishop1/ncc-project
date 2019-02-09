from __future__ import print_function
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import torchvision
from torchvision import transforms, datasets, models

def train_model(model, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    softmax = nn.LogSoftmax(dim=1)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs - 1))
        print('-' * 10)

        for i,phase in enumerate(['train','valid']):
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)
            
            running_loss = 0.0
            running_corrects = 0

            for it,data in enumerate(dataloaders[i]):

                inputs,labels = data

                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs,labels = Variable(inputs),Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = F.nll_loss(softmax(outputs),labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data).float()

            epoch_loss = running_loss / len(dataloaders[i].dataset)
            epoch_acc = running_corrects / len(dataloaders[i].dataset)

            print('{} Loss: {:4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':

    learning_rate = 0.00001
    training_iterations = 50

    simple_transforms = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    train = datasets.ImageFolder('./dataset_fine-grained/train/',simple_transforms)
    valid = datasets.ImageFolder('./dataset_fine-grained/valid/',simple_transforms)


    train_data_gen = torch.utils.data.DataLoader(train,batch_size = 32,num_workers=1)
    valid_data_gen = torch.utils.data.DataLoader(valid,batch_size = 32,num_workers=1)
    dataloaders = [train_data_gen,valid_data_gen]

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,len(train.classes))

    if torch.cuda.is_available():
        model_ft = model_ft.cuda()

    optimizer_ft = optim.SGD(model_ft.parameters(),lr = learning_rate,momentum = 0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)

    train_model(model_ft, optimizer_ft, exp_lr_scheduler, training_iterations)

def imshow(inp):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std * inp + mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    plt.show()


