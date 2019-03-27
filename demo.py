from __future__ import print_function
import torch
import os
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import torchvision
from torchvision import transforms, datasets, models
from cnn import MyNetwork, imshow


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    image_dataset = datasets.ImageFolder('dataset_fine-grained/test',data_transforms)
            
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=16, shuffle=True)

    test_images, test_labels = next(iter(dataloader))

    test_images, test_labels = test_images.to(device), test_labels.to(device)

    class_names = image_dataset.classes

    model_ft = MyNetwork(len(class_names))
    
    model_ft = model_ft.to(device)    

    model_ft.load_state_dict(torch.load('./model.pt',map_location='cpu'))
    
    model_ft.eval()

    test_images, test_labels = next(iter(dataloader))

    test_images, test_labels = test_images.to(device), test_labels.to(device)

    outputs = model_ft(test_images)
    values, test_preds = torch.max(outputs, 1)

    plt.figure(figsize=(4, 4))
    for i in range(16):
        ax = plt.subplot(4, 4, i)

        ax.axis('off')

        color = '#335599' if test_preds[i] == test_labels[i] else '#ee4433'
    
        plt.title("{} {:2.0f}% ({})".format(class_names[test_preds[i]],
                                  100*values[i],
                                  class_names[test_labels[i]]),
                                  color=color)

        imshow(test_images.cpu(0).data[i])

    plt.savefig('demo.png')
