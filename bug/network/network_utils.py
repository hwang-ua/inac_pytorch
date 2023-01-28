import torch.nn as nn

def layer_init_xavier(layer, bias=True):
    nn.init.xavier_uniform_(layer.weight)
    if bias:
        nn.init.constant_(layer.bias.data, 0)
    return layer
