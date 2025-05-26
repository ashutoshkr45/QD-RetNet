import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from tqdm import tqdm
import numpy as np
from models_qat import load_quant_separate_model_I, load_quant_separate_model_II
from models_qat.resnet import QuantizedMLP
from models import save_dml_models
from utils import AverageMeter, load_config, splitprint, runid_checker, predict_dataloader
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

    print("\ncls\t\tsen\t\tspe\t\tf1\t\tauc\t\tmap\t\tacc")
    for lbl in label2disease:
        print("{cls}\t\t{sen:.4f}\t\t{spe:.4f}\t\t{f1:.4f}\t\t{auc:.4f}\t\t{ap:.4f}\t\t{acc:.4f}".format(cls=lbl,
                                                                                    sen=results[lbl]['sen'],
                                                                                    spe=results[lbl]['spe'],
                                                                                    f1=results[lbl]['f1_score'],
                                                                                    auc=results[lbl]['auc'],
                                                                                    ap=results[lbl]['ap'], 
                                                                                    acc=results[lbl]['acc']))
    print("overall\t\t{sen:.4f}\t\t{spe:.4f}\t\t{f1:.4f}\t\t{auc:.4f}\t\t"
          "{map:.4f}\t\t{acc:.4f}\n".format(
        sen=results["overall"]["sen"],
        spe=results["overall"]["spe"],
        f1=results["overall"]["f1_score"],
        auc=results["overall"]["auc"],
        map=results["overall"]["map"],
        acc=results["overall"]["acc"]))

    return results["overall"]["map"]


def adjust_learning_rate(optimizer, optim_params):
    optim_params['lr'] *= 0.75
    print('learning rate:', optim_params['lr'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = optim_params['lr']
    if optim_params['lr'] < optim_params['lr_min']:
        return True
    else:
        return False
    

class MutualTrainer:
    def __init__(self, configs, opts, device):
        self.configs = configs
        self.opts = opts
        self.device = device

        # Initialize models - Model 1: Fundus(Teacher) -> OCT Student
        self.model1_teacher, self.model1_student = load_quant_separate_model_I(configs, device, opts.checkpoint_f)

        # Model 2: OCT(Teacher) -> Fundus(Student)
        self.model2_teacher, self.model2_student = load_quant_separate_model_II(configs, device, opts.checkpoint_o)

        # Initialize quantized alignment layers
        self.align_layer1 = QuantizedMLP().to(device)
        self.align_layer2 = QuantizedMLP().to(device)
        self.align_layer1.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
        self.align_layer2.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
        torch.ao.quantization.prepare_qat(self.align_layer1, inplace=True)
        torch.ao.quantization.prepare_qat(self.align_layer2, inplace=True)

        # Use DataParallel for multi-GPU training
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training!")
            self.model1_student = nn.DataParallel(self.model1_student)
            self.model2_student = nn.DataParallel(self.model2_student)
            self.align_layer1 = nn.DataParallel(self.align_layer1)
            self.align_layer2 = nn.DataParallel(self.align_layer2)

        # Setup optimizers
        self.optimizer_params1 = configs.train_params["sgd"]
        self.optimizer_params2 = configs.train_params["sgd"]

        self.optimizer1_student = SGD([
                                    {'params': self.model1_student.parameters()},
                                    {'params': self.align_layer1.parameters()}], 
                                    lr=self.optimizer_params1["lr"],
                                    momentum=self.optimizer_params1["momentum"],
                                    weight_decay=self.optimizer_params1["weight_decay"])
        
        self.optimizer2_student = SGD([
                                    {'params': self.model2_student.parameters()},
                                    {'params': self.align_layer2.parameters()}], 
                                    lr=self.optimizer_params2["lr"],
                                    momentum=self.optimizer_params2["momentum"],
                                    weight_decay=self.optimizer_params2["weight_decay"])

        self.optimizer1_teacher = SGD(
                                    self.model1_teacher.parameters(), 
                                    lr=self.optimizer_params1["lr"],
                                    momentum=self.optimizer_params1["momentum"],
                                    weight_decay=self.optimizer_params1["weight_decay"])
        self.optimizer2_teacher = SGD(
                                    self.model2_teacher.parameters(), 
                                    lr=self.optimizer_params2["lr"],
                                    momentum=self.optimizer_params2["momentum"],
                                    weight_decay=self.optimizer_params2["weight_decay"])
        
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        self.tolerance1 = 0
        self.tolerance2 = 0
        # Best metrics tracking
        self.best_metric1 = 0
        self.best_metric2 = 0


    def train(self, train_loader_I, test_loader_I, train_loader_II, test_loader_II):
        for epoch in tqdm(range(self.configs.train_params["max_epoch"]), desc="Training Progress"):
            print(f"\nEpoch {epoch+1}/{self.configs.train_params['max_epoch']}")
            
            # Training Model 1
            print("Training Model 1 (Fundus->OCT)")
            metric1 = self.train_model1(epoch, train_loader_I, test_loader_I)
            
            # Training Model 2
            print("Training Model 2 (OCT->Fundus)")
            metric2 = self.train_model2(epoch, train_loader_II, test_loader_II)
            
            # Save best models
            if metric1 > self.best_metric1:
                self.best_metric1 = metric1
                print(f"Saving Model 1 (Fundus->OCT) with metric: {metric1:.4f}")
                self.save_model(self.model1_student, 1, epoch, metric1)
            #elif epoch > self.optimizer_params1["lr_decay_start"]:
            #    self.tolerance1 += 1
            #    if self.tolerance1 % self.optimizer_params1["tolerance_iter_num"] == 0:
            #        if_stop = adjust_learning_rate(self.optimizer1_student, self.optimizer_params1)
            #        print("best:", self.best_metric1)
            #        if if_stop:
            #            print("Stopping due to model1")
            #            break
                
            if metric2 > self.best_metric2:
                self.best_metric2 = metric2
                print(f"Saving Model 2 (OCT->Fundus) with metric: {metric2:.4f}")
                self.save_model(self.model2_student, 2, epoch, metric2)
            #elif epoch > self.optimizer_params2["lr_decay_start"]:
            #    self.tolerance2 += 1
            #    if self.tolerance2 % self.optimizer_params2["tolerance_iter_num"] == 0:
            #        if_stop = adjust_learning_rate(self.optimizer2_student, self.optimizer_params2)
            #        print("best:", self.best_metric2)
            #        if if_stop:
            #            print("Stopping due to model2")
            #            break
                
        print(f"\nBest metrics - Model 1: {self.best_metric1:.4f}, Model 2: {self.best_metric2:.4f}")


    def train_model1(self, epoch, train_loader, test_loader):
        self.model1_teacher.train()
        self.model1_student.train()

        losses = AverageMeter()

    
        for i, (inputs, labels_onehot, _) in enumerate(train_loader):
            fundus_input = inputs[0].to(self.device)
            oct_input = inputs[1].to(self.device)
            labels_fundus = labels_onehot[1].float().to(self.device)
            labels_oct = labels_onehot[0].float().to(self.device)
            
            # Zero gradients
            self.optimizer1_student.zero_grad()
            self.optimizer1_teacher.zero_grad()
            
            # Forward passes
            preds_t1, feats_t1 = self.model1_teacher(fundus_input)
            preds_s1, feats_s1 = self.model1_student(oct_input)

            with torch.no_grad():
                preds_t_d, feats_t_d = self.model1_teacher(fundus_input)

            # Original Knowledge Distillation loss
            kd_loss = self.compute_kd_loss(epoch, self.model1_teacher, self.model1_student,
                                          feats_t_d, feats_s1, preds_t_d, preds_s1, 
                                          self.align_layer1, labels_fundus, labels_oct)
            
            # Classification losses
            cls_loss_student = self.criterion(preds_s1, labels_oct)
            cls_loss_teacher = self.criterion(preds_t1, labels_fundus)
            
            # Total loss
            total_loss = cls_loss_student + kd_loss
            
            # Backward passes
            total_loss.backward()
            cls_loss_teacher.backward()
            
            # Update weights
            self.optimizer1_student.step()
            self.optimizer1_teacher.step()
            
            losses.update(total_loss.item(), oct_input.size(0))
            
            if i % self.opts.print_freq == 0:
                print(
                    f'Model1 - Batch: [{i}\t/\t{len(train_loader)}]\t\t'
                    f'Loss: {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'KD Loss: {kd_loss:.4f}'
                )
        
        # Validation
        self.model1_student.eval()
        metric = validate(self.model1_student, test_loader, self.configs.train_params["best_metric"],
                        self.device, self.configs.cls_num, self.configs.net_name, not self.configs.if_syn)
        return metric
    

    def train_model2(self, epoch, train_loader, test_loader):
        self.model2_teacher.train()
        self.model2_student.train()
        
        losses = AverageMeter()
        

        for i, (inputs, labels_onehot, _) in enumerate(train_loader):
            fundus_input = inputs[0].to(self.device)
            oct_input = inputs[1].to(self.device)
            labels_fundus = labels_onehot[0].float().to(self.device)
            labels_oct = labels_onehot[1].float().to(self.device)
            
            # Zero gradients
            self.optimizer2_student.zero_grad()
            self.optimizer2_teacher.zero_grad()
            
            # Forward passes
            preds_t2, feats_t2 = self.model2_teacher(oct_input)
            preds_s2, feats_s2 = self.model2_student(fundus_input)
            
            with torch.no_grad():
                preds_t2_d, feats_t2_d = self.model2_teacher(oct_input)

            # Original Knowledge Distillation loss
            kd_loss = self.compute_kd_loss(epoch, self.model2_teacher, self.model2_student,
                                          feats_t2_d, feats_s2, preds_t2_d, preds_s2,
                                          self.align_layer2, labels_oct, labels_fundus)
            
            # Classification losses
            cls_loss_student = self.criterion(preds_s2, labels_fundus)
            cls_loss_teacher = self.criterion(preds_t2, labels_oct)
            
            # Total loss
            total_loss = cls_loss_student + kd_loss 
            
            # Backward passes
            total_loss.backward()
            cls_loss_teacher.backward()
            
            # Update weights
            self.optimizer2_student.step()
            self.optimizer2_teacher.step()
            
            losses.update(total_loss.item(), fundus_input.size(0))
            
            if i % self.opts.print_freq == 0:
                print(
                    f'Model2 - Batch: [{i}\t/\t{len(train_loader)}]\t\t'
                    f'Loss: {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'KD Loss: {kd_loss:.4f}'
                )
        
        # Validation
        self.model2_student.eval()
        metric = validate(self.model2_student, test_loader, self.configs.train_params["best_metric"],
                        self.device, self.configs.cls_num, self.configs.net_name, not self.configs.if_syn)
        return metric
    

    def compute_kd_loss(self, epoch, teacher, student, teacher_feats, student_feats, teacher_preds, student_preds, align_layer, labels_t, labels_s):
        """Compute Knowledge Distillation loss based on CPM and CSA"""
        if epoch < self.opts.distill_epoch:
            return 0.0
        
        feat_size = student_feats.size(-1)
        p_size = student_preds.size(-1)
            
        # Align student features
        student_feats = align_layer(student_feats)

        class_pro_t, class_pro_s = [], []
        preds_t_ens, preds_s_ens = [], []
        
        for k in range(self.configs.cls_num):
            prototype_t = torch.zeros(feat_size, dtype=torch.float).to(self.device)
            prototype_s = torch.zeros(feat_size, dtype=torch.float).to(self.device)
            pred_t_ens = torch.zeros(p_size, dtype=torch.float).to(self.device)
            pred_s_ens = torch.zeros(p_size, dtype=torch.float).to(self.device)
            total_num_t, total_num_s = 0, 0
        
            # Compute class prototypes for each class
            for batch_id in range(len(labels_s)):
                if labels_s[batch_id][k]:
                   prototype_s += student_feats[batch_id]
                   pred_s_ens += student_preds[batch_id]
                   total_num_s += 1

                if labels_t[batch_id][k]:
                    prototype_t += teacher_feats[batch_id]
                    pred_t_ens += teacher_preds[batch_id]
                    total_num_t += 1
        
            if total_num_s > 0 and total_num_t > 0:
                class_pro_t.append(torch.div(prototype_t, total_num_t))
                class_pro_s.append(torch.div(prototype_s, total_num_s))
                preds_t_ens.append(torch.div(pred_t_ens, total_num_t))
                preds_s_ens.append(torch.div(pred_s_ens, total_num_s))

        if len(class_pro_s) > 0:
            class_pro_t = torch.stack(class_pro_t, 0).to(self.device)
            class_pro_s = torch.stack(class_pro_s, 0).to(self.device)
        
            # Split features into major and minor components
            class_mean = torch.mean(class_pro_s, dim=0)
            threshold = torch.mean(class_mean)
            mask_major = torch.where(class_mean > threshold)[0]
            mask_minor = torch.where(class_mean <= threshold)[0]
        
            # Get major and minor components
            class_pro_t_major = class_pro_t[:, mask_major]
            class_pro_s_major = class_pro_s[:, mask_major]
            class_pro_t_minor = class_pro_t[:, mask_minor]
            class_pro_s_minor = class_pro_s[:, mask_minor]

        
        loss_distill_proto = loss_fn_kd(class_pro_s_major, class_pro_t_major, self.opts.temperature, self.opts.alpha) + loss_fn_kd(
                        class_pro_s_minor, class_pro_t_minor, self.opts.temperature, self.opts.alpha)
        
        preds_s_ens = torch.stack(preds_s_ens, 0).to(self.device)
        preds_t_ens = torch.stack(preds_t_ens, 0).to(self.device)

        # compute similarity matrix
        s_sim = torch.cosine_similarity(preds_s_ens.unsqueeze(1), preds_s_ens.unsqueeze(0), dim=-1)
        t_sim = torch.cosine_similarity(preds_t_ens.unsqueeze(1), preds_t_ens.unsqueeze(0), dim=-1)

        loss_distill_sim = loss_fn_kd(s_sim, t_sim, self.opts.temperature, self.opts.beta)
        
        return loss_distill_proto + loss_distill_sim

    
    def save_model(self, model, model_num, epoch, metric):
        """Save quantized model"""
        model.eval()
        quantized_model = torch.ao.quantization.convert(model.cpu())
        save_dml_models(
            quantized_model.state_dict(), 
            self.opts, 
            epoch, 
            metric,
            if_syn=self.configs.if_syn,
            best_model=True,
            model_num=model_num
        )
        model = model.to(self.device)
