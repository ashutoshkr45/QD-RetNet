import numpy as np
import torch
import os
import torch.nn as nn
from . import resnet

def init_resnet50(num_classes=4, pretrained=True, heatmap=False, early=False):
    model = resnet.resnet50(pretrained=pretrained, heatmap=heatmap)
    model.avgpool.kernel_size = 14
    # fc_inchannel = model.fc.in_features
    # model.fc = nn.Linear(fc_inchannel, num_classes)
    return model

def load_quant_separate_model_I(configs, device, checkpoint_f=None):
    """Modified model loader to create quantization-aware models"""
    use_gpu = "cpu" != device.type
    
    # Initialize teacher model (model_f)
    if checkpoint_f is None:
        model_f = init_resnet50(pretrained=True, heatmap=configs.heatmap, num_classes=configs.cls_num)
    else:
        model_f = init_resnet50(num_classes=configs.cls_num, pretrained=False, heatmap=configs.heatmap)
        if use_gpu:
            model_f = model_f.to(device)
            model_f.load_state_dict(torch.load(checkpoint_f, map_location=device))
            model_f = nn.DataParallel(model_f)
        else:
            model_f.load_state_dict(torch.load(checkpoint_f, map_location={"cpu"}))
    
    # Initialize student model (model_o)
    model_o = init_resnet50(pretrained=True, heatmap=configs.heatmap, num_classes=configs.cls_num)
    
    # Set quantization configurations
    model_o.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    torch.ao.quantization.prepare_qat(model_o, inplace=True)
    
    # Move models to appropriate device
    if use_gpu:
        model_f = model_f.to(device)
        model_o = model_o.to(device)

    return model_f, model_o


def load_quant_separate_model_II(configs, device, checkpoint_o=None):
    use_gpu = "cpu" != device.type

    if checkpoint_o is None:
        model_o = init_resnet50(pretrained=True, heatmap=configs.heatmap, num_classes=configs.cls_num)
    else:
        model_o = init_resnet50(num_classes=configs.cls_num, pretrained=False, heatmap=configs.heatmap)
        if use_gpu:
            model_o = model_o.to(device)
            model_o.load_state_dict(torch.load(checkpoint_o, map_location=device))
            model_o = nn.DataParallel(model_o)
        else:
            model_o.load_state_dict(torch.load(checkpoint_o, map_location={"cpu"}))

    model_f = init_resnet50(pretrained=True, heatmap=configs.heatmap, num_classes=configs.cls_num)

    # Set quantization configurations
    model_f.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    torch.ao.quantization.prepare_qat(model_f, inplace=True)

    if use_gpu:
        model_o = model_o.to(device)
        model_f = model_f.to(device)

    return model_o, model_f


def save_model(model_state, opts, epoch, best_metric, save_filename=None, best_model=False, best_epoch=-1,if_syn=False):
    rootpath = opts.train_collection
    if if_syn:
        rootpath = opts.syn_collection
    valset_name = os.path.split(opts.val_collection)[-1]
    config_filename = opts.model_configs
    teacher_student = opts.teacher + "-->" + opts.student
    run_id = opts.run_id
    path = os.path.join(rootpath, "models", valset_name, config_filename, teacher_student, "run_" + str(run_id))

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
    torch.save(model_state, save_path)
    print(f"Model saved at {save_path}")