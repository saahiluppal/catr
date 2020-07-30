import torch

from models import caption
from configuration import Config

dependencies = ['torch', 'torchvision']

def resnet101_catr(pretrained=False):
    config = Config()
    model, _ = caption.build_model(config)
    
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url='https://github.com/saahiluppal/catr/releases/download/0.1/weights_9348032.pth',
            map_location='cpu'
        )
        model.load_state_dict(checkpoint['model'])
    
    return model