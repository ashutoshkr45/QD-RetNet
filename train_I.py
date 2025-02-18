import os
import sys
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import importlib
import numpy as np
from models_qat import load_quant_separate_model_I
from models_qat.resnet import QuantizedMLP
from models import save_model
from data.DataLoaders import MultiDataLoader_I
from utils import AverageMeter, load_config, splitprint, runid_checker, predict_dataloader, setup_logging
import random
from metrics import multilabel_confusion_matrix, accuracy_score, spe_score, sen_score, f1_score
from metrics import confusion_matrix as cfm
from sklearn.metrics import roc_curve, auc, average_precision_score

label2disease = ['NOR', 'AMD', 'WAMD', 'DR', 'CSC', 'PED', 'MEM', 'FLD', 'EXU', 'CNV', 'RVO']


def loss_fn_kd(outputs, teacher_outputs, T, alpha):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return KD_loss


def parse_args():
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--train_collection", type=str,
                        default='image_data/topcon-mm/train',
                        help="train collection path")
    parser.add_argument("--val_collection", type=str,
                        default='image_data/topcon-mm/val',
                        help="val collection path")
    parser.add_argument("--test_collection", type=str,
                        default='image_data/topcon-mm/test',
                        help="test collection path")
    parser.add_argument("--print_freq", default=30, type=int, help="print frequent (default: 30)")
    parser.add_argument("--model_configs", type=str, default='config.py',
                        help="filename of the model configuration file.")
    parser.add_argument("--model_id", default="q1", type=str, help="to make unique path for saving model")
    parser.add_argument("--run_id", default=0, type=int, help="run_id (default: 0)")
    parser.add_argument("--device", default="1", type=str, help="cuda:n or cpu (default: 0)")
    parser.add_argument("--num_workers", default=0, type=int, help="number of threads for sampling. (default: 0)")
    parser.add_argument("--overwrite", default=True, type=bool, help="overwrite existing files")
    parser.add_argument("--checkpoint", default="image_data/topcon-mm/train/models/val/config_fundus.py_s1/run_0/best_model.pth",
                         type=str, help="fundus model checkpoint path")
    parser.add_argument("--batch_size", default=8, type=int, help="size of a batch")
    parser.add_argument("--distill_epoch", default=0, type=float, help="epoch to start distillation")
    parser.add_argument("--seed", default=100, type=int)
    parser.add_argument("--temperature", default=4, type=float)
    parser.add_argument("--alpha", default=2, type=float)
    parser.add_argument("--beta", default=1, type=float)
    args = parser.parse_args()
    return args


def validate(model, val_loader, selected_metric, device, cls_num, net_name="mm-model", verbose=True):
    if verbose:
        print("-" * 45 + "validation" + "-" * 45)
    predicts, scores, expects, predicts_fine, scores_fine = predict_dataloader(model, val_loader, device,
                                                                                             net_name,
                                                                                             if_test=False)
    predicts = np.array(predicts)
    scores = np.array(scores)
    expects = np.array(expects)

    results = {'overall': {}}
    for lb in label2disease:
        results[lb] = {}

    confusion_matrix = multilabel_confusion_matrix(expects, predicts)
    results['overall']['cm'] = confusion_matrix
    for i in range(cls_num):
        results[label2disease[i]]['spe'] = spe_score(confusion_matrix[i])
        results[label2disease[i]]['sen'] = sen_score(confusion_matrix[i])
        results[label2disease[i]]['f1_score'] = f1_score(results[label2disease[i]]['spe'],
                                                         results[label2disease[i]]['sen'])
        results[label2disease[i]]['acc'] = accuracy_score(confusion_matrix[i])

        predicts_specific = scores[:, i].tolist()
        expects_specific = expects[:, i].tolist()
        fpr, tpr, th = roc_curve(expects_specific, predicts_specific, pos_label=1)
        auc_specific = auc(fpr, tpr)
        results[label2disease[i]]['auc'] = auc_specific
        results[label2disease[i]]['ap'] = average_precision_score(expects_specific, predicts_specific)

    results["overall"]["sen"] = np.average([results[cls_name]["sen"] for cls_name in label2disease])
    results["overall"]["spe"] = np.average([results[cls_name]["spe"] for cls_name in label2disease])
    results["overall"]["f1_score"] = np.average([results[cls_name]["f1_score"] for cls_name in label2disease])
    results["overall"]["auc"] = np.average([results[cls_name]["auc"] for cls_name in label2disease])
    results["overall"]["map"] = np.average([results[cls_name]["ap"] for cls_name in label2disease])
    results["overall"]["acc"] = np.average([results[cls_name]["acc"] for cls_name in label2disease])

    print("\ncls\tsen\tspe\tf1\tauc\tmap\tacc")
    for lbl in label2disease:
        print("{cls}\t{sen:.4f}\t{spe:.4f}\t{f1:.4f}\t{auc:.4f}\t{ap:.4f}\t{acc:.4f}".format(cls=lbl,
                                                                                    sen=results[lbl]['sen'],
                                                                                    spe=results[lbl]['spe'],
                                                                                    f1=results[lbl]['f1_score'],
                                                                                    auc=results[lbl]['auc'],
                                                                                    ap=results[lbl]['ap'], 
                                                                                    acc=results[lbl]['acc']))
    print("overall\t{sen:.4f}\t{spe:.4f}\t{f1:.4f}\t{auc:.4f}\t"
          "{map:.4f}\t{acc:.4f}\n".format(
        sen=results["overall"]["sen"],
        spe=results["overall"]["spe"],
        f1=results["overall"]["f1_score"],
        auc=results["overall"]["auc"],
        map=results["overall"]["map"],
        acc=results["overall"]["acc"]),
        "confusion matrix:\n {}".format(results["overall"]["cm"]))

    return results["overall"]["map"]


def adjust_learning_rate(optimizer, optim_params):
    optim_params['lr'] *= 0.5
    print('learning rate:', optim_params['lr'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']
    if optim_params['lr'] < optim_params['lr_min']:
        return True
    else:
        return False

    
def main(opts):
    # Set up logging
    log_filename = setup_logging()
    print(f"Logging training information to: {log_filename}\n")

    # load model configs
    configs = load_config(opts.model_configs)

    if not runid_checker(opts, configs.if_syn):
        return
    splitprint()
    
    device = torch.device("cuda" if (torch.cuda.is_available() and opts.device != "cpu") else "cpu")

    # get trainset and valset dataloaders for training
    data_initializer = MultiDataLoader_I(opts, configs)
    train_loader, val_loader, test_loader = data_initializer.get_training_dataloader()

    # load model
    splitprint()
    checkpoint = opts.checkpoint
    model_f, model_o = load_quant_separate_model_I(configs, device, checkpoint)

    # Initialize quantization-aware MLP
    align_layer = QuantizedMLP().to(device)
    align_layer.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    torch.ao.quantization.prepare_qat(align_layer, inplace=True)

    criterion = torch.nn.BCEWithLogitsLoss()

    # Optimizer setup
    optimizer_params = configs.train_params["sgd"]
    optimizer = torch.optim.SGD([{'params': model_o.parameters()},
                               {'params': align_layer.parameters()}],
                               lr=optimizer_params["lr"],
                               momentum=optimizer_params["momentum"],
                               weight_decay=optimizer_params["weight_decay"])
    
    optimizer_f = torch.optim.SGD(model_f.parameters(),
                                lr=optimizer_params["lr"],
                                momentum=optimizer_params["momentum"],
                                weight_decay=optimizer_params["weight_decay"])
    
    # Training loop
    best_metric = 0
    best_model_wts = copy.deepcopy(model_o.state_dict())
    
    for epoch in range(configs.train_params["max_epoch"]):
        print(f'Epoch {epoch + 1}/{configs.train_params["max_epoch"]}')
        
        # Training phase
        model_f.train()
        model_o.train()

        # Initialize meters for monitoring training progress
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        
        for i, (inputs, labels_onehot, imagenames) in enumerate(train_loader):
            labels_onehot_o = labels_onehot[0].float().to(device)
            labels_onehot_f = labels_onehot[1].float().to(device)
            
            optimizer.zero_grad()
            optimizer_f.zero_grad()
            
            # Forward passes
            preds_o, feats_o = model_o(inputs[1].to(device))
            preds_f, feats_f = model_f(inputs[0].to(device))
            
            # Knowledge distillation logic
            # Get sizes for distillation calculations
            inputs_size = inputs[1].size(0)
            p_size = preds_o.size(-1)
            feat_size = feats_o.size(-1)

            loss_distill, loss_distill_proto, loss_distill_sim = 0.0, 0.0, 0.0
            
            if epoch >= opts.distill_epoch:
                class_pro_f_d, class_pro_o, weight_cls = [], [], []
                preds_f_d_ens, preds_o_ens = [], []
                with torch.no_grad():
                    preds_f_d, feats_f_d = model_f(inputs[0].to(device))
                    p_o = torch.nn.Sigmoid()(preds_o)

                # align feature vector
                feats_o = align_layer(feats_o)

                # compute class prototype
                for k in range(configs.cls_num):
                    prototype_o = torch.zeros(feat_size, dtype=torch.float).to(device)
                    prototype_f = torch.zeros(feat_size, dtype=torch.float).to(device)
                    pred_o_ens = torch.zeros(p_size, dtype=torch.float).to(device)
                    pred_f_d_ens = torch.zeros(p_size, dtype=torch.float).to(device)
                    total_num_f, total_num_o = 0, 0
                    for batch_id in range(len(labels_onehot_o)):
                        if labels_onehot_o[batch_id][k]:
                            prototype_o += feats_o[batch_id]
                            pred_o_ens += preds_o[batch_id]
                            total_num_o += 1

                        if labels_onehot_f[batch_id][k]:
                            prototype_f += feats_f_d[batch_id]
                            pred_f_d_ens += preds_f_d[batch_id]
                            total_num_f += 1

                    if total_num_o > 0 and total_num_f > 0:
                        class_pro_o.append(torch.div(prototype_o, total_num_o))
                        class_pro_f_d.append(torch.div(prototype_f, total_num_f))
                        preds_o_ens.append(torch.div(pred_o_ens, total_num_o))
                        preds_f_d_ens.append(torch.div(pred_f_d_ens, total_num_o))

                if len(class_pro_o) > 0: 
                    class_pro_o = torch.stack(class_pro_o, 0).to(device)
                    class_pro_f_d = torch.stack(class_pro_f_d, 0).to(device)

                    class_mean = torch.mean(class_pro_o, dim=0)
                    threshold = torch.mean(class_mean)
                    mask_major = torch.where(class_mean > threshold)[0]
                    mask_minor = torch.where(class_mean <= threshold)[0]

                    class_pro_o_major = class_pro_o[:, mask_major]
                    class_pro_f_d_major = class_pro_f_d[:, mask_major]
                    class_pro_o_minor = class_pro_o[:, mask_minor]
                    class_pro_f_d_minor = class_pro_f_d[:, mask_minor]

                    loss_distill_proto = loss_fn_kd(class_pro_o_major, class_pro_f_d_major, opts.temperature, opts.alpha) + loss_fn_kd(
                        class_pro_o_minor, class_pro_f_d_minor, opts.temperature, opts.alpha)

                    preds_o_ens = torch.stack(preds_o_ens, 0).to(device)
                    preds_f_d_ens = torch.stack(preds_f_d_ens, 0).to(device)

                    # compute similarity matrix
                    o_sim = torch.cosine_similarity(preds_o_ens.unsqueeze(1), preds_o_ens.unsqueeze(0), dim=-1)
                    f_sim = torch.cosine_similarity(preds_f_d_ens.unsqueeze(1), preds_f_d_ens.unsqueeze(0), dim=-1)

                    loss_distill_sim = loss_fn_kd(o_sim, f_sim, opts.temperature, opts.beta)
                    loss_distill = loss_distill_proto + loss_distill_sim
            
            # Calculate losses
            loss_fundus = criterion(preds_f, labels_onehot_f)
            loss_cls = criterion(preds_o, labels_onehot_o)
            loss = loss_cls + loss_distill
            
            # Backward passes
            loss.backward()
            loss_fundus.backward()
            
            optimizer.step()
            optimizer_f.step()

             # Update metrics
            losses.update(loss.item(), inputs_size)
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Print progress
            if i % opts.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss:.4f} (classification: {loss_cls:.4f}, distill: {loss_distill:.4f}; '
                      '{loss_distill_proto:.4f}, {loss_distill_sim:.4f})\t'
                    .format(
                        epoch, i, len(train_loader),
                        batch_time=batch_time, data_time=data_time, loss=loss,
                        loss_cls=loss_cls, loss_distill=loss_distill, 
                        loss_distill_proto=loss_distill_proto, 
                        loss_distill_sim=loss_distill_sim
                    ))
        
        # Validation phase
        model_o.eval()
        test_metric = validate(model_o, test_loader, configs.train_params["best_metric"],
                             device, configs.cls_num, configs.net_name, not configs.if_syn)
        
        # Save best model
        if test_metric > best_metric:
            best_metric = test_metric
            best_model_wts = copy.deepcopy(model_o.state_dict())
            print(f"Save better weights, metric value: {best_metric}")
            
            # Convert and save quantized model
            model_o.eval()
            quantized_model = torch.ao.quantization.convert(model_o.cpu())
            save_model(quantized_model.state_dict(), opts, epoch, best_metric, if_syn=configs.if_syn, best_model=True)
            model_o = model_o.to(device)
    
    print(f"Best validation metric: {best_metric}")

    print(f"\nTraining completed. Final best validation metric: {best_metric}")
    print(f"All training logs have been saved to: {log_filename}")


if __name__ == "__main__":
    opts = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.device
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts.seed)
    main(opts)
