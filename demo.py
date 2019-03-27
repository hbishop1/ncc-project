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
from cnn import MyNetwork

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = '#335599' if predicted_label == true_label else '#ee4433'
    
    print(predictions_array.data)
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('#ee4433')
    thisplot[true_label].set_color('#335599')

def visualize_model(model):

    test_images, test_labels = next(iter(dataloader))

    test_images, test_labels = test_images.to(device), test_labels.to(device)

    outputs = model(test_images)
    _, test_preds = torch.max(outputs, 1)
    
    num_rows = 4
    num_cols = 4
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, test_preds.cpu(), test_labels.cpu(), test_images.cpu().squeeze().permute(1,3,2,0).contiguous().permute(3,2,1,0))
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, test_preds.cpu(), test_labels.cpu())
    plt.savefig('demo.png')

if __name__ == '__main__':


    data_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    image_dataset = datasets.ImageFolder('dataset_fine-grained/test',data_transforms)
            
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=16, shuffle=True)

    class_names = image_dataset.classes

    model_ft = MyNetwork(len(class_names))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_ft = model_ft.to(device)    

    model_ft.load_state_dict(torch.load('./model.pt',map_location='cpu'))
    
    model_ft.eval()

    visualize_model(model_ft)