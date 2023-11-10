import os
import random

import lightning.pytorch as pl
import numpy as np
import torch
def set_seed(seed=10):
    pl.seed_everything(seed)
    np.random.seed(seed=seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.set_float32_matmul_precision('medium')
