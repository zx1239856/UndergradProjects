import argparse
import logging
import torch
import random
import numpy as np
import os


def init_logger(log_dir, log_file):
    logger = logging.getLogger()
    format_str = '[%(asctime)s %(filename)s#%(lineno)3d] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt='%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, log_file))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_arguments():
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--ctx-weight', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume-step', action='store_true')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')

    # optimizer
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--base-lr', type=float, default=1e-5)
    parser.add_argument('--lr-decay-step', type=int, default=99999999)
    parser.add_argument('--lr-decay-rate', type=float, default=0.9)

    # io
    parser.add_argument('--train-split', type=str, default='../split/train.npy')
    parser.add_argument('--val-split', type=str, default='../split/val.npy')
    parser.add_argument('--model-dir', type=str, default='../models/model')
    parser.add_argument('--save-dir', type=str, default=None)

    args = parser.parse_args()
    return args
