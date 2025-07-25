from collections import defaultdict

from .utils import *
from .dataset import MultiDataset_fo, MultiDataset_of, SingleDataset
import random
import numpy as np
import cv2 as cv
import torch
from . import augmentation


class FundusBuffer:
    def __init__(self, img_f_path_list, img_o_path_list, labels_f_list=None, labels_o_list=None,
                 aug_params=None, transform=None, cls_num=4):
        self.cls_num = cls_num

        self.img_f_path_list = img_f_path_list
        self.img_o_path_list = img_o_path_list
        self.labels_f_list = labels_f_list
        self.labels_list = labels_o_list

        self.aug_params = aug_params
        self.transform = transform

        self.class_list = defaultdict(list)
        for idx in range(len(self.labels_f_list)):
            for c in range(len(self.labels_f_list[idx])):
                if labels_f_list[idx][c] == 1:
                    self.class_list[c].append(img_f_path_list[idx])

    def get_fundus_data(self, label_o, queue_size=5):
        label_all = [0] * self.cls_num
        for l in label_o:
            for idx_l in range(len(l)):
                if l[idx_l]:
                    label_all[idx_l] = 1
        img_ensemble = []
        for idx_l in range(len(label_all)):
            if label_all[idx_l]:
                img_f_d_path = random.sample(self.class_list[idx_l], min(queue_size, len(self.class_list[idx_l])))
                img_f_d = [(cv.imread(img_path) / 255.).astype(np.float32) for img_path in img_f_d_path]

                if self.aug_params:
                    aug = augmentation.OurAug(self.aug_params)
                    img_f_d = [aug.process(img) for img in img_f_d]
                if self.transform:
                    img_f_d = [self.transform(img) for img in img_f_d]
                img_f_d = torch.stack(img_f_d, 0)
            else:
                img_f_d = None

            img_ensemble.append(img_f_d)

        return img_ensemble


class DataOrganizer_fo:
    @staticmethod
    def get_fundus_data(collection, image_path, aug_params=None, transform=None, if_test=False, cls_num=4, if_eval=False):
        path_file_f = os.path.join(collection, "cfp.txt")
        imgs_f_path_list, labels_f_list = load_pathfile(image_path, "fundus_images", path_file_f, "cfp")

        return SingleDataset(imgs_f_path_list, labels_f_list, aug_params, transform, if_test, cls_num)

    @staticmethod
    def get_oct_data(collection, image_path, aug_params=None, transform=None, if_test=False, cls_num=4, if_eval=False):
        path_file = os.path.join(collection, "oct.txt")
        imgs_path_list, labels_list = load_pathfile(image_path, "oct_images", path_file, "oct")
        return SingleDataset(imgs_path_list, labels_list, aug_params, transform, if_test, cls_num)

    @staticmethod
    def get_mm_data(
            collection, image_path, loosepair=False, aug_params=None, transform=None, if_test=False, cls_num=4, if_eval=False,
            if_syn=False, syn_collection=None):
        path_file_f = os.path.join(collection, "cfp.txt")
        imgs_f_path_list, labels_f_list = load_pathfile(image_path, "fundus_images", path_file_f, "cfp")

        path_file = os.path.join(collection, "oct.txt")
        imgs_o_path_list, labels_o_list = load_pathfile(image_path, "oct_images", path_file, "oct")

        if if_eval:
            fundus_buffer = None
        else:
            fundus_buffer = FundusBuffer(imgs_f_path_list, imgs_o_path_list, labels_f_list, labels_o_list, aug_params,
                                         transform, cls_num)

        return MultiDataset_fo(imgs_f_path_list, imgs_o_path_list, labels_f_list, labels_o_list, aug_params, transform,
                            if_test, cls_num, if_eval)
    
class OCTBuffer:
    """Buffer class for storing and retrieving OCT images for knowledge distillation"""
    def __init__(self, img_f_path_list, img_o_path_list, labels_f_list=None, labels_o_list=None,
                 aug_params=None, transform=None, cls_num=4):
        self.cls_num = cls_num
        self.img_o_path_list = img_o_path_list
        self.labels_o_list = labels_o_list
        self.aug_params = aug_params
        self.transform = transform

        self.class_list = defaultdict(list)
        for idx in range(len(self.labels_o_list)):
            for c in range(len(self.labels_o_list[idx])):
                if labels_o_list[idx][c] == 1:
                    self.class_list[c].append(img_o_path_list[idx])

    def get_oct_data(self, label_f, queue_size=5):
        """Get relevant OCT images based on Fundus image label"""
        label_all = [0] * self.cls_num
        for l in label_f:
            for idx_l in range(len(l)):
                if l[idx_l]:
                    label_all[idx_l] = 1
                    
        img_ensemble = []
        for idx_l in range(len(label_all)):
            if label_all[idx_l]:
                img_o_path = random.sample(self.class_list[idx_l], min(queue_size, len(self.class_list[idx_l])))
                img_o = [(cv.imread(img_path) / 255.).astype(np.float32) for img_path in img_o_path]

                if self.aug_params:
                    aug = augmentation.OurAug(self.aug_params)
                    img_o = [aug.process(img) for img in img_o]
                if self.transform:
                    img_o = [self.transform(img) for img in img_o]
                img_o = torch.stack(img_o, 0)
            else:
                img_o = None

            img_ensemble.append(img_o)

        return img_ensemble
    
class DataOrganizer_of:
    @staticmethod
    def get_mm_data(
            collection, image_path, loosepair=False, aug_params=None, transform=None, 
            if_test=False, cls_num=4, if_eval=False, if_syn=False, syn_collection=None):
        """Get multimodal data with Fundus as student and OCT as teacher"""
        path_file_f = os.path.join(collection, "cfp.txt")
        imgs_f_path_list, labels_f_list = load_pathfile(image_path, "fundus_images", path_file_f, "cfp")

        path_file_o = os.path.join(collection, "oct.txt")
        imgs_o_path_list, labels_o_list = load_pathfile(image_path, "oct_images", path_file_o, "oct")

        if if_eval:
            oct_buffer = None
        else:
            oct_buffer = OCTBuffer(imgs_f_path_list, imgs_o_path_list, labels_f_list, labels_o_list, 
                                 aug_params, transform, cls_num)

        return MultiDataset_of(imgs_f_path_list, imgs_o_path_list, labels_f_list, labels_o_list, 
                               aug_params, transform, if_test, cls_num, if_eval)
