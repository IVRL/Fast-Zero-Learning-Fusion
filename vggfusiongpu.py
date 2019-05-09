import math
import numbers
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.vgg import vgg19

def get_activation(model, layer_number, input_image):
    out = input_image
    for i in range(layer_number):
        with torch.no_grad():
            out = model.features[i](out)
    l1out = torch.sum(torch.abs(out), dim=1, keepdim=True)
    l1out = (l1out - l1out.min()) / (l1out.max() - l1out.min())
    return l1out

def softmax(feats, with_exp=False):
    feats = torch.cat(feats, dim=1)
    if with_exp:
        feats = torch.exp(feats)
    feats = feats / (feats.sum(dim=1, keepdim=True) + 1e-5)
    h,w = feats.shape[2:]
    return feats

def to_pytorch(image):
    np_input = image.astype(np.float32) / 255.
    if np_input.ndim == 2:
        np_input = np.repeat(np_input[None, None], 3, axis=1)
    else:
        np_input = np.transpose(np_input, (2, 0, 1))[None]
    return torch.from_numpy(np_input).cuda()

def fuse(inputs, model=None, with_exp=False, layer_number=2):   
    with torch.no_grad():
        if model is None:
            model = vgg19(True)
        model.cuda().eval()

        tc_inputs = []
        relus_acts = []
        for input_img in inputs:
            tc_input = to_pytorch(input_img)
            relus_act = get_activation(model, layer_number, tc_input)

            tc_inputs.append(tc_input)
            relus_acts.append(relus_act)

        return F.interpolate(softmax(relus_acts, with_exp), size=tc_inputs[0].shape[2:])