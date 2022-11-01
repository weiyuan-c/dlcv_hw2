import argparse
import os
import glob
import random
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from models import DANN

# setting seed
seed = 72373981
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class ImgDataset(Dataset):
    def __init__(self, path, tfm=None, files=None):
        super(ImgDataset).__init__()
        self.path = path
        self.files = sorted(glob.glob(os.path.join(path, '*')))
        if files != None:
            self.files = files
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        img = Image.open(file_name).convert('RGB')
        img = self.transform(img)

        return img



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument('-t', '--test_dir', type=str, default='hw2_data/digits/svhn/test')
    parser.add_argument('-c', '--csv_dir', type=str, default='prediction.csv')
    # hyper-parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=28)
    cfg = parser.parse_args()

    # data transfomation
    test_tfm = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    # load model
    model = DANN().to(device)
    # dataset
    val_set = ImgDataset(cfg.test_dir, tfm=test_tfm)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    # load weight
    if 'svhn' in cfg.test_dir:
        # if data is from svhn dataset
        print('mnist-m -> svhn')
        model.load_state_dict(torch.load('svhn.ckpt', map_location=device))
    else:
        print('mnist-m -> usps')
        # if model is trained on usps or the dataset can't be identified
        model.load_state_dict(torch.load('usps.ckpt', map_location=device))

    # validation 
    model.eval()
    val_pred = []
    for i, img in enumerate(tqdm(val_loader)):
        img = img.to(device)
        with torch.no_grad():
            output, _ = model(img, 0)
            pred = list(output.argmax(dim=1).squeeze().detach().cpu().numpy())
        val_pred += pred

    df = pd.DataFrame()
    ids = sorted([x.split('/')[-1] for x in glob.glob(os.path.join(cfg.test_dir, '*.png'))])
    df['img_name'] = np.array(ids)
    df['label'] = val_pred
    df.to_csv(cfg.csv_dir, index=False)
