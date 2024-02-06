import torch
from copy import deepcopy
from dynamic_layers import MaskedConv2d, MaskedMLP

def print_layer_keep_ratio(model):
    total = 0.
    keep = 0.
    layer_keep = []
    # print(model.modules())
    for layer in model.modules():
        if isinstance(layer, MaskedMLP):
            total, keep, layer_keep = count_mlp(layer, total, keep, layer_keep)
        elif isinstance(layer, MaskedConv2d):
            total, keep, layer_keep = count_conv(layer, total, keep, layer_keep)
    return keep / total, layer_keep


def count_mlp(layer, total, keep, layer_keep):
    abs_weight = torch.abs(layer.weight)
    threshold = layer.threshold.view(abs_weight.shape[0], -1)
    abs_weight = abs_weight - threshold
    mask = layer.step(abs_weight)
    ratio = torch.sum(mask) / mask.numel()
    total += mask.numel()
    keep += torch.sum(mask)
    # logger.info("Layer threshold {:.4f}".format(layer.threshold[0]))
    layer_ratio = "{}, keep ratio {:.4f}".format(layer, ratio)
    print(layer_ratio)
    layer_keep.append(layer_ratio)
    return total, keep, layer_keep


def count_conv(layer, total, keep, layer_keep):
    weight_shape = layer.weight.shape
    threshold = layer.threshold.view(weight_shape[0], -1)
    weight = torch.abs(layer.weight)
    weight = weight.view(weight_shape[0], -1)
    weight = weight - threshold
    mask = layer.step(weight)
    # print(mask)
    ratio = torch.sum(mask) / mask.numel()
    total += mask.numel()
    keep += torch.sum(mask)
    layer_ratio = "{}, keep ratio {:.4f}".format(layer, ratio)
    print(layer_ratio)
    layer_keep.append(layer_ratio)
    return total, keep, layer_keep


def set_training_mode(model, training_mode):
    for layer in model.modules():
        if isinstance(layer, MaskedMLP) or isinstance(layer, MaskedConv2d):
            layer.sparse_train = training_mode


def sparsity_regularizer(model, lambda_):
    sr_loss = 0
    for layer in model.modules():
        if isinstance(layer, MaskedConv2d) or isinstance(layer, MaskedMLP):
            sr_loss += lambda_ * torch.sum(torch.exp(-layer.threshold))

    return sr_loss

def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)