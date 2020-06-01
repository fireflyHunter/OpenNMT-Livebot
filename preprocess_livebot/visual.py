import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models



def build_resnet():
    model = models.resnet50(pretrained=True)
    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def get_resnet_feature(filename, model):

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    return output.flatten().data
