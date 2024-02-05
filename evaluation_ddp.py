import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import numpy as np
import faulthandler
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from loader import Loader
from utils.logger import Logger
from utils.utils import AverageMeter, AverageMeterForDict, str2bool, distributed_mean


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="val", type=str, help="Mode, train/val/test")
    parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Val batch size")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")
    parser.add_argument("--adv_cfg_path", required=True, default="", type=str)
    parser.add_argument("--model_path", required=False, type=str, help="path to the saved model")
    return parser.parse_args()


def main():
    args = parse_arguments()
    faulthandler.enable()

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()

    is_main = True if local_rank == 0 else False

    # logger only for print
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "log/" + date_str
    logger = Logger(date_str=date_str, enable=is_main, log_dir=log_dir,
                    enable_flags={'writer': False, 'mailbot': False})

    loader = Loader(args, device, is_ddp=True, world_size=world_size, local_rank=local_rank, verbose=is_main)
    logger.print('[Resume] Loading state_dict from {}'.format(args.model_path))
    loader.set_resmue(args.model_path)
    (train_set, val_set), net, _, _, evaluator = loader.load()

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    dl_train = DataLoader(train_set,
                          batch_size=args.train_batch_size,
                          num_workers=48,
                          collate_fn=train_set.collate_fn,
                          drop_last=False,
                          sampler=train_sampler,
                          pin_memory=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    dl_val = DataLoader(val_set,
                        batch_size=args.val_batch_size,
                        num_workers=48,
                        collate_fn=val_set.collate_fn,
                        drop_last=False,
                        sampler=val_sampler,
                        pin_memory=True)

    net.eval()
    with torch.no_grad():
        # # * Train
        # dl_train.sampler.set_epoch(0)
        # train_start = time.time()
        # train_eval_meter = AverageMeterForDict()
        # for i, data in enumerate(tqdm(dl_train, disable=(not is_main))):
        #     out = net(data)
        #     post_out = net.module.post_process(out)

        #     eval_out = evaluator.evaluate(post_out, data)
        #     train_eval_meter.update(eval_out, n=data['BATCH_SIZE'])

        # # make eval results into a Tensor
        # eval_res = [elem.avg for k, elem in train_eval_meter.metrics.items()]
        # eval_res = torch.from_numpy(np.array(eval_res)).to(device)

        # train_eval_mean = distributed_mean(eval_res)
        # train_eval = dict()
        # for i, key in enumerate(list(train_eval_meter.metrics.keys())):
        #     train_eval[key] = train_eval_mean[i].item()
        # train_eval_meter.reset()
        # train_eval_meter.update(train_eval)

        # dist.barrier()  # sync
        # logger.print('\nTraining set finish, cost {} secs'.format(time.time() - train_start))
        # logger.print('-- ' + train_eval_meter.get_info())

        # * Validation
        dl_val.sampler.set_epoch(0)
        val_start = time.time()
        val_eval_meter = AverageMeterForDict()
        for i, data in enumerate(tqdm(dl_val, disable=(not is_main))):
            data_in = net.module.pre_process(data)
            out = net(data_in)
            post_out = net.module.post_process(out)

            eval_out = evaluator.evaluate(post_out, data)
            val_eval_meter.update(eval_out, n=data['BATCH_SIZE'])

        # make eval results into a Tensor
        eval_res = [elem.avg for k, elem in val_eval_meter.metrics.items()]
        eval_res = torch.from_numpy(np.array(eval_res)).to(device)

        val_eval_mean = distributed_mean(eval_res)
        val_eval = dict()
        for i, key in enumerate(list(val_eval_meter.metrics.keys())):
            val_eval[key] = val_eval_mean[i].item()
        val_eval_meter.reset()
        val_eval_meter.update(val_eval)

        dist.barrier()  # sync
        logger.print('\nValidation set finish, cost {:.2f} secs'.format(time.time() - val_start))
        logger.print('-- ' + val_eval_meter.get_info())

    dist.destroy_process_group()
    logger.print('\nExit...')


if __name__ == "__main__":
    main()
