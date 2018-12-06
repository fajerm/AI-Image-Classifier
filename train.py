import argparse

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


from main import load_data, build_model, validation, train_model, accuracy_test, save_checkpoint, process_image, imshow, predict, bar_chart, predict_names

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Neural Network')
    parser.add_argument('data_dir', default = 'flowers',  help = 'Enter the directory containing training and testing data.')
    parser.add_argument('--save_dir',  default = 'saved_model.pth', help = 'Enter a directory to save the model')
    parser.add_argument('--arch',  default = 'vgg16', help = 'Enter the pretrained model to use, defaults to VGG16')
    parser.add_argument('--learning_rate',  default = .001, help = 'Learning rate to train the model')
    parser.add_argument('--dropout', default = 0.2,
                    help = 'Enter dropout for training the model, default is 0.2.')
    parser.add_argument('--hidden_units',  default = 4096, help = 'Number of hidden layer units')
    parser.add_argument('--epochs', type = int,   default = 1, help = 'Number of epochs to be used to train the model')
    parser.add_argument('--gpu', default = True, help = 'Turn GPU on or off, defaults to on')

    results = parser.parse_args()
    
    data_dir = results.data_dir
    save_dir = results.save_dir
    droppercentage = results.dropout
    arch = results.arch
    LR = results.learning_rate
    hidden_units = results.hidden_units
    epochs = results.epochs
    gpu = results.gpu

    
    train_set, valid_set, test_set, train_loader, valid_loader, test_loader = load_data(data_dir)
    
    pretrained_model = results.arch
    pretrained_model = getattr(models, pretrained_model)(pretrained=True)
    
    input_layer = pretrained_model.classifier[0].in_features
    
    
    model, hls = build_model(arch, gpu, input_layer, droppercentage)
    
    
    model, criterion, optimizer = train_model(model, train_set, valid_set, train_loader, valid_loader, epochs, LR, gpu)

    accuracy = accuracy_test(model, test_loader)
    
    print ("Accuracy of your trained model: {:.2f}%".format(accuracy*100))
    
    save_checkpoint(model, hls, epochs, droppercentage, train_set, optimizer, criterion, save_dir)
    print('Model has been saved!')
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    