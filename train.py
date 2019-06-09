# Imports here
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import OrderedDict
import argparse
import os

def args_paser():
    paser = argparse.ArgumentParser(description='trainer file')
    paser.add_argument('--data_dir', type=str,
                       default='flowers', help='dataset directory')
    paser.add_argument('--gpu', type=bool, default='True',
                       help='True: gpu, False: cpu')
    paser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    paser.add_argument('--epochs', type=int, default=1, help='num of epochs')
    paser.add_argument('--arch', type=str, default='dens161',
                       help='architecture')
    paser.add_argument('--hidden_units', type=int,
                       default=1000, help='hidden units for layer')
    paser.add_argument('--save_dir', type=str,
                       default='checkpoint.pth', help='save train model to a file')
    args = paser.parse_args()
    return args

def process_data(train_dir, test_dir, validation_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_data = datasets.ImageFolder(validation_dir, transform=validation_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader( train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
    
    return trainloader, testloader, validationloader


def basic_model(arch):
    # Load pretrained_network
    if arch == None or arch == 'dense161':
        load_model = models.densenet161(pretrained=True)
        print('Use dense161')
    else:
        print('Please dense net models, defaulting to dense161')
        load_model = models.densenet161(pretrained=True)

    # Freeze parameters
    for param in load_model.parameters():
        param.requires_grad = False
    return load_model


def set_classifier(model, hidden_units):
    if hidden_units == None:
        hidden_units = 1000
    input = model.classifier.in_features
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(1000, 500)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc3', nn.Linear(500, 250)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc4', nn.Linear(250, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    return model


def train_model(epochs, trainloader, validationloader, gpu, model, optimizer, criterion):
    if type(epochs) == type(None):
        epochs = 10
        print("Epochs = 10")

    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)

    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 50

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validationloader):.3f}")
                running_loss = 0
                model.train()
    return model


def testing_model(model, testloader, gpu, criterion):
    testing_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            if gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                model.to('cuda')

            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            testing_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Testing loss: {testing_loss/len(testloader):.3f}..")
    print(f"Testing accuracy: {accuracy/len(testloader):.3f}")
    model.train()


def save_checkpoint(model, epochs, optimizer, train_dir, save_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'arch': 'densenet161',
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'epochs': epochs,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    return torch.save(checkpoint, save_dir)


def main():
    #get and parse args
    args = args_paser()
    #setting data_dir
    data_dir = args.data_dir
    #building dir paths to data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    #getting dataloader object for each dataset
    trainloaders, testloaders, validationloaders = process_data(
        train_dir, test_dir, valid_dir)
    #constructing model
    model = basic_model(args.arch)
    model = set_classifier(model, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    #training the model
    trmodel = train_model(args.epochs, trainloaders,
                          validationloaders, args.gpu, model, optimizer, criterion)
    # testing the model
    testing_model(trmodel, testloaders, args.gpu, criterion)
    #saving the model to a chekpoint file
    save_checkpoint(trmodel, args.epochs, optimizer,
                    train_dir, args.save_dir)

    print('Completed!')


if __name__ == '__main__':
    main()
