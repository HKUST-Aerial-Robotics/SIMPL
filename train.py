import os
import sys
import time
import subprocess
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import faulthandler
from tqdm import tqdm
#
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
#
from loader import Loader
from utils.logger import Logger
from utils.utils import AverageMeterForDict
from utils.utils import save_ckpt, set_seed


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", type=str, help="Mode, train/val/test")
    parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Val batch size")
    parser.add_argument("--train_epoches", type=int, default=10, help="Number of epoches for training")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation intervals")
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--logger_writer", action="store_true", help="Enable tensorboard")
    parser.add_argument("--adv_cfg_path", required=True, default="", type=str)
    parser.add_argument("--rank_metric", required=False, type=str, default="brier_fde_k", help="Ranking metric")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--no_pbar", action="store_true", help="Hide progress bar")
    parser.add_argument("--model_path", required=False, type=str, help="path to the saved model")
    return parser.parse_args()


def main():
    args = parse_arguments()

    faulthandler.enable()
    start_time = time.time()
    set_seed(args.seed)

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "log/" + date_str
    logger = Logger(date_str=date_str, log_dir=log_dir, enable_flags={'writer': args.logger_writer})
    # log basic info
    logger.log_basics(args=args, datetime=date_str)

    loader = Loader(args, device, is_ddp=False)
    if args.resume:
        logger.print('[Resume] Loading state_dict from {}'.format(args.model_path))
        loader.set_resmue(args.model_path)
    (train_set, val_set), net, loss_fn, optimizer, evaluator = loader.load()

    dl_train = DataLoader(train_set,
                          batch_size=args.train_batch_size,
                          shuffle=True,
                          num_workers=8,
                          collate_fn=train_set.collate_fn,
                          drop_last=True,
                          pin_memory=True)
    dl_val = DataLoader(val_set,
                        batch_size=args.val_batch_size,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=val_set.collate_fn,
                        drop_last=True,
                        pin_memory=True)

    niter = 0
    best_metric = 1e6
    rank_metric = args.rank_metric
    net_name = loader.network_name()

    for epoch in range(args.train_epoches):
        logger.print('\nEpoch {}'.format(epoch))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # * Train
        epoch_start = time.time()
        train_loss_meter = AverageMeterForDict()
        train_eval_meter = AverageMeterForDict()
        net.train()
        for i, data in enumerate(tqdm(dl_train, disable=args.no_pbar, ncols=80)):
            data_in = net.pre_process(data)
            out = net(data_in)
            loss_out = loss_fn(out, data)

            post_out = net.post_process(out)
            eval_out = evaluator.evaluate(post_out, data)

            optimizer.zero_grad()
            loss_out['loss'].backward()
            lr = optimizer.step()

            train_loss_meter.update(loss_out)
            train_eval_meter.update(eval_out)
            niter += args.train_batch_size
            logger.add_dict(loss_out, niter, prefix='train/')

        # print('epoch: {}, lr: {}'.format(epoch, lr))
        optimizer.step_scheduler()
        max_memory = torch.cuda.max_memory_allocated(device=device) // 2 ** 20

        loss_avg = train_loss_meter.metrics['loss'].avg
        logger.print('[Training] Avg. loss: {:.6}, time cost: {:.3} mins, lr: {:.3}, peak mem: {} MB'.
                     format(loss_avg, (time.time() - epoch_start) / 60.0, lr, max_memory))
        logger.print('-- ' + train_eval_meter.get_info())

        logger.add_scalar('train/lr', lr, it=epoch)
        logger.add_scalar('train/max_mem', max_memory, it=epoch)
        for key, elem in train_eval_meter.metrics.items():
            logger.add_scalar(title='train/{}'.format(key), value=elem.avg, it=epoch)

        if ((epoch + 1) % args.val_interval == 0) or epoch > int(args.train_epoches / 2):
            # * Validation
            with torch.no_grad():
                val_start = time.time()
                val_loss_meter = AverageMeterForDict()
                val_eval_meter = AverageMeterForDict()
                net.eval()
                for i, data in enumerate(tqdm(dl_val, disable=args.no_pbar, ncols=80)):
                    data_in = net.pre_process(data)
                    out = net(data_in)
                    loss_out = loss_fn(out, data)

                    post_out = net.post_process(out)
                    eval_out = evaluator.evaluate(post_out, data)

                    val_loss_meter.update(loss_out)
                    val_eval_meter.update(eval_out)

                logger.print('[Validation] Avg. loss: {:.6}, time cost: {:.3} mins'.format(
                    val_loss_meter.metrics['loss'].avg, (time.time() - val_start) / 60.0))
                logger.print('-- ' + val_eval_meter.get_info())

                for key, elem in val_loss_meter.metrics.items():
                    logger.add_scalar(title='val/{}'.format(key), value=elem.avg, it=epoch)
                for key, elem in val_eval_meter.metrics.items():
                    logger.add_scalar(title='val/{}'.format(key), value=elem.avg, it=epoch)

                if (epoch >= args.train_epoches / 2):
                    if val_eval_meter.metrics[rank_metric].avg < best_metric:
                        model_name = date_str + '_{}_best.tar'.format(net_name)
                        save_ckpt(net, optimizer, epoch, 'saved_models/', model_name)
                        best_metric = val_eval_meter.metrics[rank_metric].avg
                        print('Save the model: {}, {}: {:.4}, epoch: {}'.format(
                            model_name, rank_metric, best_metric, epoch))

        if int(100 * epoch / args.train_epoches) in [20, 40, 60, 80]:
            model_name = date_str + '_{}_ckpt_epoch{}.tar'.format(net_name, epoch)
            save_ckpt(net, optimizer, epoch, 'saved_models/', model_name)
            logger.print('Save the model to {}'.format('saved_models/' + model_name))

    logger.print("\nTraining completed in {:.2f} mins".format((time.time() - start_time) / 60.0))

    # save trained model
    model_name = date_str + '_{}_epoch{}.tar'.format(net_name, args.train_epoches)
    save_ckpt(net, optimizer, epoch, 'saved_models/', model_name)
    print('Save the model to {}'.format('saved_models/' + model_name))

    print('\nExit...\n')


if __name__ == "__main__":
    main()
