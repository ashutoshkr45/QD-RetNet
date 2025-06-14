import numpy as np
import torch
import os
import torch.nn as nn
from . import resnet
from . import multimodal_resnet

def init_resnet50(num_classes=4, pretrained=True, heatmap=False, early=False):
    model = resnet.resnet50(pretrained=pretrained, heatmap=heatmap)
    model.avgpool.kernel_size = 14
    # fc_inchannel = model.fc.in_features
    # model.fc = nn.Linear(fc_inchannel, num_classes)
    return model


def load_single_stream_model(configs, device, checkpoint=None, early=False):
    use_gpu = "cpu" != device.type
    if checkpoint:
        model = init_resnet50(pretrained=False, heatmap=configs.heatmap, num_classes=configs.cls_num, early=early)
        print("load checkpoint '{}'".format(checkpoint))
        if use_gpu:
            model = model.to(device)
            model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
        else:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    else:
        model = init_resnet50(pretrained=True, heatmap=configs.heatmap, num_classes=configs.cls_num, early=early)
        if use_gpu:
            model = model.to(device)

    return model


def load_separate_model_I(configs, device, checkpoint_f=None):
    use_gpu = "cpu" != device.type
    if checkpoint_f is None:
        model_f = init_resnet50(pretrained=True, heatmap=configs.heatmap, num_classes=configs.cls_num)
    else:
        model_f = init_resnet50(num_classes=configs.cls_num, pretrained=False, heatmap=configs.heatmap)
        if use_gpu:
            model_f = model_f.to(device)
            model_f.load_state_dict(torch.load(checkpoint_f, map_location="cuda"))
        else:
            model_f.load_state_dict(torch.load(checkpoint_f, map_location={"cpu"}))

    model_o = init_resnet50(pretrained=True, heatmap=configs.heatmap, num_classes=configs.cls_num)

    if use_gpu:
        model_f = model_f.to(device)
        model_o = model_o.to(device)

    return model_f, model_o

def load_separate_model_II(configs, device, checkpoint_o=None):
    use_gpu = "cpu" != device.type
    if checkpoint_o is None:
        model_o = init_resnet50(pretrained=True, heatmap=configs.heatmap, num_classes=configs.cls_num)
    else:
        model_o = init_resnet50(num_classes=configs.cls_num, pretrained=False, heatmap=configs.heatmap)
        if use_gpu:
            model_o = model_o.to(device)
            model_o.load_state_dict(torch.load(checkpoint_o, map_location="cuda"))
        else:
            model_o.load_state_dict(torch.load(checkpoint_o, map_location={"cpu"}))

    model_f = init_resnet50(pretrained=True, heatmap=configs.heatmap, num_classes=configs.cls_num)

    if use_gpu:
        model_o = model_o.to(device)
        model_f = model_f.to(device)

    return model_o, model_f


def save_model(model_state, opts, epoch, best_metric, save_filename=None, best_model=False, best_epoch=-1,
               if_syn=False):
    rootpath = opts.train_collection
    if if_syn:
        rootpath = opts.syn_collection
    valset_name = os.path.split(opts.val_collection)[-1]
    config_filename = opts.model_configs + "_" + opts.model_id
    run_id = opts.run_id
    path = os.path.join(rootpath, "models", valset_name, config_filename, "run_" + str(run_id))

    # Ensure the directory exists, if not create it
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created.")

    if save_filename is None:
        if best_model:
            save_filename = "best_model.pth"
        else:
            save_filename = "last_model.pth"
    torch.save(model_state, os.path.join(path, save_filename))


def save_dml_models(model_state, opts, epoch, best_metric, save_filename=None, best_model=False, best_epoch=-1, if_syn=False, model_num=0):
    rootpath = opts.train_collection
    if if_syn:
        rootpath = opts.syn_collection
    valset_name = os.path.split(opts.val_collection)[-1]
    config_filename = opts.model_configs
    run_id = opts.run_id
    path = os.path.join(rootpath, "models", valset_name, config_filename, "run_" + str(run_id) + "_model_" + str(model_num))

    # Ensure the directory exists, if not create it
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created.")

    if save_filename is None:
        if best_model:
            save_filename = "best_model.pth"
        else:
            save_filename = "last_model.pth"

    # Save the model
    save_path = os.path.join(path, save_filename)
    
    if os.path.exists(save_path) and not opts.overwrite:
        print(f"Model already exists at {save_path} and overwrite is set to False. Skipping save.")
    else:
        torch.save(model_state, save_path)
        print(f"Model saved at {save_path}")
