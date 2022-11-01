import argparse
import os
import random
import torch
import torchvision
import torchvision.transforms as T
from torch.autograd import Variable
import numpy as np
from models import generator

# setting seed
seed = 9686468
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # setting path
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--weight', type=str, default='G.ckpt')
    # setting hyper parameters
    cfg = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    G = generator().to(device)
    G.load_state_dict(torch.load(cfg.weight, map_location=device))
    G.eval()
    sample_num = 1000
    test_sample = Variable(torch.randn(sample_num, 100)).to(device)
    output = (G(test_sample) + 1.0) / 2.0
    for i in range(sample_num):
        torchvision.utils.save_image(output[i], os.path.join(cfg.test_dir, '%s.png'%(i+1)))
    
