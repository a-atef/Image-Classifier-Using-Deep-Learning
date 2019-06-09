import torch
import json
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
    paser = argparse.ArgumentParser(description='predict file')
    paser.add_argument('--input', type=str, help='image path')
    paser.add_argument('--checkpoint', type=str,
                       default='checkpoint.pth', help='path to checkpoint file')
    paser.add_argument('--top_k', type=int, default=5,
                       help='return top k classes')
    paser.add_argument('--category_names', type=str, default='cat_to_name.json',
                       help='category names')
    paser.add_argument('--gpu', type=bool, default='True',
                       help='True: gpu, False: cpu')
    args = paser.parse_args()
    return args


def load_checkpoint(filepath):
    checkpoint = torch.load(
        filepath, map_location=lambda storage, loc: storage)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    pil_image = Image.open(image)
    pil_image = preprocess(pil_image).float()
    np_image = np.array(pil_image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image


def predict(image_path, model, cat_to_name, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = torch.FloatTensor(image)

    image.unsqueeze_(0)
    if gpu == True:
        image = image.to('cuda')
        model.to('cuda')

    logp = model(image)
    logp = logp.cpu()

    ps = torch.exp(logp)

    ps.max(dim=1)
    probs, top_classes = ps.topk(topk)

    #Converting the indices to class
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    prob_arr = probs.data.numpy()[0]
    pred_indexes = top_classes.data.numpy()[0].tolist()
    pred_labels = [idx_to_class[x] for x in pred_indexes]
    pred_class = [cat_to_name[str(x)] for x in pred_labels]
    model.train()
    return prob_arr, pred_class

def main():
    args = args_paser()
    
    model = load_checkpoint(args.checkpoint)
    model.eval()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    print(predict(args.input, model, cat_to_name, args.gpu, args.top_k))
    
    print('Completed!')


if __name__ == '__main__':
    main()
