import imp
import os
import subprocess
import sys
import datetime
import logging
import torch
import numpy as np
#
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, date_str, enable=True, log_dir='', enable_flags={'writer': True}):
        self.enable = enable
        self.enable_flags = enable_flags
        for k, v in self.enable_flags.items():
            self.enable_flags[k] = v and self.enable

        self.date_str = date_str
        self.print('[Logger] {} - logger enable_flags: {}'.format(self.date_str, self.enable_flags))

        if log_dir == '':
            self.log_dir = "log/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.log_dir = log_dir

        if self.enable_flags['writer']:
            self.writer = SummaryWriter(self.log_dir)

            filename = 'log/log_{}.log'.format(self.date_str)
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(filename=filename,
                                format='%(asctime)s:%(message)s', level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S')

    def log_basics(self, args, datetime):
        # log basic info
        self.print("Args: {}\n".format(args))
        _, ret = subprocess.getstatusoutput("echo $PWD")
        self.print("Project Path: {}".format(ret))
        self.print("Datetime: {}\n".format(datetime))
        _, ret = subprocess.getstatusoutput("git log -n 1")
        self.print("Commit Msg: {}\n".format(ret))

        self.print("======================================\n")

    def add_scalar(self, title, value, it):
        if self.enable_flags['writer']:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            self.writer.add_scalar(title, value, it)

    def add_dict(self, data, it, prefix=''):
        if self.enable_flags['writer']:
            for key, val in data.items():
                title = prefix + key
                self.add_scalar(title, val, it)

    def print(self, info):
        if self.enable:
            print(info)
            logging.info(info)
