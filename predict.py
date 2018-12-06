import argparse

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

import json

from main import load_data, build_model, validation, train_model, accuracy_test, save_checkpoint, load_model, process_image, imshow, predict, bar_chart, predict_names


parser = argparse.ArgumentParser(description='Use your Neural Network to make prediction on image.')

parser.add_argument('--image_path', action='store',
                    default = '../aipnd-project/flowers/test/10/image_07117.jpg',
                    help='Enter path to image you want to classify.')

parser.add_argument('--save_dir', action='store',
                    dest='save_directory', default = 'saved_model.pth',
                    help='Enter location to save checkpoint in.')

parser.add_argument('--arch',  default = 'vgg16', help = 'Enter the pretrained model to use, defaults to VGG16')

parser.add_argument('--top_k', action='store',
                    dest='topk', type=int, default = 3,
                    help='Enter number of top most likely classes to view, default is 3.')

parser.add_argument('--cat_to_name', action='store',
                    dest='cat_name_dir', default = 'cat_to_name.json',
                    help='Enter path to image.')

parser.add_argument('--gpu', action="store_true", default=True,
                    help='Turn GPU mode on or off, defaults to on.')

results = parser.parse_args()

save_dir = results.save_directory
image_path = results.image_path
top_k = results.topk
gpu = results.gpu
cat_names = results.cat_name_dir
arch = results.arch

with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

pretrained_model = results.arch
model = getattr(models, pretrained_model)(pretrained=True)
input_layer = model.classifier[0].in_features

loaded_model, class_to_idx = load_model(save_dir, input_layer, gpu, arch)

probs, classes = predict(image_path, loaded_model, class_to_idx, top_k)

print(probs[0])
print(classes) 

# Print name of predicted flower with highest probability
x = probs[0]
y = [cat_to_name[str(i)] for i in classes]

print("-- Probablities :")
a=0    
for a in range(0, top_k, 1):
    print("Image Class : {}  \nProbablity : {:.2f}%\n".format(y[a],x[a]*100))
    a += 1

print("This image is most likely a {}".format(y[0]))






