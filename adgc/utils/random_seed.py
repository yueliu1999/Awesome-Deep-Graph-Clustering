# -*- coding: utf-8 -*-
# @Author  : Yue Liu
# @Email   : yueliu19990731@163.com
# @Time    : 2022/9/21 0:49
import torch
import random
import numpy as np


def setup_seed(seed):
    """
    fix the random seed
    :param seed: the random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None
