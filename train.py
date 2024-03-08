from torchsummary import summary

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

from transformers import AutoImageProcessor, Dinov2ForImageClassification
from torch import nn


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    print("training is started")

    torch.save(model, os.path.join('./checkpoints', 'best_model_params.pt'))
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print("Starting phase ------------------------------------------------------- ", phase)
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            iteration = 0 

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
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if (iteration % 100 == 0):
                    print("iteration:", iteration, iteration / len(dataloaders[phase]) * 100)
                iteration+=1
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                torch.save(model, os.path.join('./checkpoints', '{}_pretrained.pt'.format(epoch)))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model, os.path.join('./checkpoints', '{}_best_model_params.pt'.format(epoch)))

        

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model

if __name__ == "__main__":


    
    print("loading dataset")

    # dataloader
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = '/vision/group/ImageNet_2012'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                shuffle=True)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Fine tune the model.
    print("loading model")
    # Load the model
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    print("train model on ", device)
    train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=3)


    # model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-small-imagenet1k-1-layer")
    # model = model.to('cuda')

    # # Freeze early layers
    # for param in model.parameters():
    #     param.requires_grad = False
    # n_inputs = model.classifier.in_features

    # n_classes = 1000
    # # Add on classifier
    # model.classifier = nn.Sequential(
    #     nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))


    # # train dataloader
    # imagenet_train = torchvision.datasets.ImageNet('/vision/u/jennyxu6/CLAE/datasets/tiny-imagenet-200', split='train')
    # train_dataloader = torch.utils.data.dataloader(imagenet_train, batch_size=4, shuffle=True)

    # trainiter = iter(train_dataloader)
    # features, labels = next(trainiter)
    # print(features.shape)
    # print(labels.shape)


    # download pre-trained backbone. 
    # dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

    
    # for param in dinov2_vits14_reg.dinov2.parameters():
    #     param.requires_grad = False

    # Download pre-trained Dionv2 classifier head
    # dinov2_vits14_reg_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')


