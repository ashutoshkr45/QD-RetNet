import os
import sys
import copy
import random
import torch
import numpy as np
import argparse
import importlib
from data.DataLoaders import MultiDataLoader_I, MultiDataLoader_II
from dml_trainer import MutualTrainer
from utils import load_config, splitprint, setup_logging


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
    parser.add_argument("--run_id", default=0, type=int, help="run_id (default: 0)")
    parser.add_argument("--device", default="1", type=str, help="cuda:n or cpu (default: 1)")
    parser.add_argument("--num_workers", default=0, type=int, help="number of threads for sampling. (default: 0)")
    parser.add_argument("--overwrite", default=True, type=bool, help="overwrite existing files")
    parser.add_argument("--checkpoint_f", default="image_data/topcon-mm/train/models/val/config_fundus.py_s1/run_0/best_model.pth",
                         type=str, help="fundus model checkpoint path")
    parser.add_argument("--checkpoint_o", default="image_data/topcon-mm/train/models/val/config_oct.py_s2/run_0/best_model.pth",
                         type=str, help="oct model checkpoint path")
    parser.add_argument("--batch_size", default=8, type=int, help="size of a batch")
    parser.add_argument("--distill_epoch", default=0, type=float, help="epoch to start distillation")
    parser.add_argument("--dml_epoch", default=1000, type=int, help="epoch to start DML")
    parser.add_argument("--seed", default=100, type=int)
    parser.add_argument("--temperature", default=4, type=float)
    parser.add_argument("--alpha", default=2, type=float)
    parser.add_argument("--beta", default=1, type=float)
    args = parser.parse_args()
    return args


def main(opts):
    # Set up logging
    log_filename = setup_logging()
    print(f"Logging training information to: {log_filename}\n")

    # load model configs
    configs = load_config(opts.model_configs)

    splitprint()
    
    device = torch.device("cuda" if (torch.cuda.is_available() and opts.device != "cpu") else "cpu")

    # get trainset and testset dataloaders 
    data_initializer1 = MultiDataLoader_I(opts, configs)
    train_loader1, val_loader1, test_loader1 = data_initializer1.get_training_dataloader()

    data_initializer2 = MultiDataLoader_II(opts, configs)
    train_loader2, val_loader2, test_loader2 = data_initializer2.get_training_dataloader()

    splitprint()

    trainer = MutualTrainer(configs, opts, device)

    trainer.train(train_loader_I=train_loader1,
                  test_loader_I=test_loader1,
                  train_loader_II=train_loader2,
                  test_loader_II=test_loader2)
    
    print(f"\nTraining completed.")
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
