

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


import json
from PIL import Image
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms

def load_data(data_dir):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Define transformations
    transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    transform_valid = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    transform_test = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])


    #Apply transforms
    train_set = datasets.ImageFolder(train_dir, transform=transform_train)
    valid_set = datasets.ImageFolder(valid_dir, transform=transform_valid)
    test_set = datasets.ImageFolder(test_dir, transform=transform_test)


    #Define dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size =32,shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 20, shuffle = True)
    
    
    return train_set, valid_set, test_set, train_loader, valid_loader, test_loader
    
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def build_model(arch, is_gpu, input_layer, Percentage_Drop):
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        input_layer = 25088
        hls = [4000, 1000]
    elif arch == 'densenet161' :
        model = models.densenet161(pretrained = True)
        input_layer = 2208
        hls = [1000, 550]
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
        input_layer = 9216
        hls = [3000, 1150]
    else: 
        raise ValueError("Choose valid name as 'vgg16', 'densenet161' or 'alexnet'.")
        
    output_layers = len(cat_to_name)
    
    for param in model.parameters():
        param.requires_grad = False
    
    Classifier_Layers = OrderedDict([
    ('Input', nn.Linear(input_layer, hls[0], bias = True)),
    ('Relu1', nn.ReLU()),
    ('Drop1', nn.Dropout(p = Percentage_Drop)),

    ('Hidden_Layer1', nn.Linear(hls[0], hls[1], bias = True)),
    ('Relu2',nn.ReLU()),
    ('Drop2', nn.Dropout(p = Percentage_Drop)),

    ('Hidden_Layer2',nn.Linear(hls[1], output_layers, bias = True)),
    ('Output', nn.LogSoftmax(dim=1))])
    
    model.classifier = nn.Sequential(Classifier_Layers)

    
    
    if is_gpu and torch.cuda.is_available():
        model.cuda()
        
    return model, hls

def validation(model, optimizer, valid_loader, criterion):
    vloss = 0
    accuracy = 0
    for ii, (inputs2, labels2) in enumerate(valid_loader):
                optimizer.zero_grad()
		if is_gpu and torch.cuda.is_available():
                	inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                model.to('cuda:0')
                with torch.no_grad():    
                    outputs = model.forward(inputs2)
                    vloss += criterion(outputs,labels2)
                    ps = torch.exp(outputs).data
                    equality = (labels2.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
    vloss = vloss / len(valid_loader)
    accuracy = accuracy /len(valid_loader) 
    return vloss, accuracy

def train_model(model, train_set, valid_set, train_loader, valid_loader, epochs, LR, is_gpu):
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), LR)
    
    if is_gpu and torch.cuda.is_available():
        model.to('cuda')    
    
    print_every = 20
    steps = 0
    
    for t in range(epochs):
        
        model.train()
    
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
        
            if is_gpu and torch.cuda.is_available():
		inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
            optimizer.zero_grad()
        
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
    
            if steps % print_every == 0:
                model.eval()
            
                vloss, accuracy = validation(model, optimizer, valid_loader, criterion)
            
                print("Epoch: {}/{}... ".format(t+1, epochs),
                    "Loss: {:.4f}".format(running_loss/print_every),
                    "Validation Loss {:.4f}".format(vloss),
                    "Accuracy: {:.4f} %".format(accuracy*100))
            
            
                running_loss = 0
                model.train()
    print("Training completed!")    
    return model, criterion, optimizer
    
    
def accuracy_test(model, test_loader):
    
    model.eval()
    correct = 0
    total = 0
    if is_gpu and torch.cuda.is_available():
    	model.to('cuda:0')
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
	    if is_gpu and torch.cuda.is_available():
            	images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy =  correct / total 
    return accuracy
    
    
def save_checkpoint(model, hidden_layers, epochs, Percentage_Drop, train_set, optimizer, criterion, save_dir):   
    
    model.class_to_idx = train_set.class_to_idx
    
    checkpoint = {'model': model,
                  'hidden_layers': hidden_layers,
                  'no_of_epochs': epochs,
                  'droppercentage': Percentage_Drop,
                  'optimizer_state': optimizer.state_dict(),
                  'criterion': criterion,
                  'class_to_idx': model.class_to_idx,
                  'model_state': model.state_dict()}

    torch.save(checkpoint, save_dir)
    
    
    
    
def load_model(path, input_layer, is_gpu, arch):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    hidden_layers = checkpoint['hidden_layers']
    Percentage_Drop = checkpoint['droppercentage']
    model_state = checkpoint['model_state']
    loaded_model, hls = build_model(arch, is_gpu, input_layer, Percentage_Drop)
    class_to_idx = checkpoint['class_to_idx']
    loaded_model.load_state_dict(model_state)
    return loaded_model, class_to_idx
    
    
    
def process_image(image):
    
    i = Image.open(image)
    transform = transforms.Compose([transforms.Resize(256),
                       transforms.CenterCrop(224),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], 
                                            [0.229, 0.224, 0.225])])
    transformed_i = transform(i)
    
    return transformed_i.numpy()    
    
    
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
   
    image = image.transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax   
    
       
def predict(image_path, model, class_to_idx, topk=5):

    model.eval()
    image = process_image(image_path)
    image = torch.from_numpy(image)
    
    image = image.unsqueeze_(0).float()
    image = image.to('cuda')
    model = model.to('cuda')
    
    output = model.forward(image)
    probabilities = torch.exp(output).data.topk(topk)
    probs = probabilities[0].tolist()
    classes = probabilities[1].tolist()

    idx_to_class = {v:k for k,v in class_to_idx.items()}
    class_idx = list()
    for i in classes[0]:
        class_idx.append(idx_to_class[i])

    return probs, class_idx



def bar_chart(cat_to_name, prob):
   
    plt.rcdefaults()
    fig,ax = plt.subplots(figsize=(10,4))

    y_pos = np.arange(len(cat_to_name))
    ax.set_xlabel('Probability of flower')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cat_to_name)
    ax.invert_yaxis() 

    ax.barh(y_pos, prob, align='center',
            color='purple', ecolor='black')


    plt.show()
    
def predict_names(model, image_path, class_to_idx, cat_to_name, topk):
    
    prob, cat = predict(image_path, model, class_to_idx, topk)
    prob = prob.cuda()[0]
    cat = cat.cuda()[0]
    cat_name = []
    for category in cat:
        cat_name.append(cat_to_name[str(category)])
    imshow(process_image(image_path))

    for i in range(topk):
        print("The image is {} with probability {:.2f}%.".format(cat_name[i], prob[i]*100))
    bar_chart(cat_to_name, prob)
    
    
    
    
    
    
    
    
    
