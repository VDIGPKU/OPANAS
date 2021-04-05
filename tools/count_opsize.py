import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         OptimizerHook, build_optimizer)
from mmdet.apis import multi_gpu_test_search, single_gpu_test_search
from mmdet.core import wrap_fp16_model
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
import numpy as np
from torch.autograd import Variable
import collections
import sys
import time
import copy
from mmdet.core import encode_mask_results, tensor2imgs
import logging
sys.setrecursionlimit(10000)
import argparse
import torch.distributed as dist

import functools
import random
import os
from mmdet.models.necks.spos_opsc import OPS

PRIMITIVES = ['TDM_dcn', 'BUM_dcn', 'PCONV_dcn', 'FSM_dcn']
def countop(paths, channel):
    opsize = 0
    fp = 0
    for path in paths:
        op = OPS[path](channel, channel, True, True)
        opsize += op.size
        fp += op.fp

    #print(opsize)
    return opsize, fp

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('log',
                        help='train log file path',
                        default='./work_dirs/faster_rcnn_r50_sposfpn3_uniform_dcn_p4st12_c64_256_1x_coco/epoch_12_ea_prun_0_20210104_075032.log')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    name = args.log
    print(os.getcwd())
    print(name)
    #name = '/data/liangtingting/projects/panas_super/work_dirs/faster_rcnn_r50_sposfpn3_uniform_dcn_p4st12_c64_256_1x_coco/epoch_12_ea_prun_0_20210104_075032.log'
    op_name = os.path.splitext(name)[0] + '.txt'

    print(op_name)
    f = open(name, 'r')
    wf = open(op_name,'w')

    for line in f:
        if '[' in line and 'AP' in line:
            st = line.index('(')
            ed = line.index(')')
            paths = str(line[st+1:ed])
            paths = paths.split(', ')
            op_paths = [int(i) for i in paths]
            channel = op_paths[-1]
            cand = [PRIMITIVES[i] for i in op_paths[:-1]]
            opsize, fp = countop(cand, channel)
            ap = line.index('AP')
            map = line[ap+3:ap+15]
            wf.write(str(cand) + ' ' + str(channel) + ' ' + map + ' ' + str(opsize) + ' ' + str(fp) + '\n')
            print(cand, channel, map, opsize, fp)
        if 'top 50 result' in line:
            break

if __name__ == '__main__':
    main()