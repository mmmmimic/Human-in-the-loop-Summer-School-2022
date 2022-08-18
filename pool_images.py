import os.path
import sys
import pandas as pd
import fungichallenge.participant as fcp
import random
import torch
import torch.nn as nn
import cv2
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, Resize
from albumentations import RandomCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, CenterCrop, PadIfNeeded, RandomResizedCrop, RandomScale
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score
from efficientnet_pytorch import EfficientNet
import numpy as np
import tqdm
from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler
import time

class NetworkFungiDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['image'].values[idx]
        label = 0
        if self.df['class'].values[idx] is not None:
            label = int(self.df['class'].values[idx])
        try:
            image = cv2.imread(file_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Fails on GPU Cluster - never stops
        except cv2.error as e:
            print("OpenCV error with", file_path, "error", e)
        except IOError:
            print("IOError with", file_path)
        except:
            print("Could not read or convert", file_path)
            print(sys.exc_info())
            return None, None

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label



def get_transforms(data):
    width = 299
    height = 299

    if data == 'train':
        return Compose([
            RandomResizedCrop(width, height, scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif data == 'valid':
        return Compose([
            Resize(width, height),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        print("Unknown data set requested")
        return None


def seed_torch(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_logger(log_file='train.log'):
    log_format = '%(asctime)s %(levelname)s %(message)s'

    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))

    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))

    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np
        
        
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2 


def get_embedding(model, x):
    x = model.extract_features(x)
    x = model._avg_pooling(x)
    x = x.flatten(-2, -1).squeeze(-1)
    return x # [B, 1280]

def get_similar_images(nw_dir, tm, tm_pw):
    data_file = os.path.join(nw_dir, "data_with_labels.csv")
    df = pd.read_csv(data_file)
    n_classes = len(df['class'].unique())
    
    pool_file = os.path.join(nw_dir, "pool_dataset.csv")
    pool_df = pd.read_csv(pool_file)

    trs = get_transforms(data='valid')
    pool_dataset = NetworkFungiDataset(pool_df, transform=trs)
    
    # batch_sz * accumulation_step = 64
    batch_sz = 16
    
    pool_loader = DataLoader(pool_dataset, batch_size=batch_sz, num_workers=8, shuffle=False)
    seed_torch(777)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, n_classes)
    model.load_state_dict(torch.load('network/DF20M-EfficientNet-B0_best_accuracy.pth'))

    model.to(device)
    
    model.eval()
    
    img_ids = np.load('img_id.npy')
    class_ids = np.load('class_id.npy')
    

    img_dict = dict(zip(list(df['image']), list(df['class'])))
    
    
    for class_id in class_ids:
        img_ids = list(filter(lambda x: img_dict[x] == class_id, img_dict.keys()))
        img_reps = []
        for img_id in img_ids:
            img = cv2.imread(img_id)
            augmented = trs(image=img)
            img = augmented['image']
            img = img.unsqueeze(0).to(device) # [1, 3 , H, W]
            x = model.extract_features(img)
            x = model._avg_pooling(x)
            img_rep = x.flatten(-2, -1).squeeze(-1)
            img_reps.append(img_rep.detach().cpu())
        
        img_rep = torch.cat(img_reps, dim=0)
        img_rep = torch.mean(img_rep, dim=0, keepdim=True) # [1, ]
        
        pool_inds = []
        pool_values = []
        for data, _ in pool_loader:
            data = data.to(device)
            data = model.extract_features(data)
            data = model._avg_pooling(data)
            pool_img_rep = data.flatten(-2, -1).squeeze(-1)
            # pool_img_rep = get_embedding(model, data)
            pool = torch.topk(torch.cosine_similarity(pool_img_rep, img_rep.to(device), dim=1), k=10)
            values = pool[0]
            inds = pool[1]
            pool_inds.append(inds.detach().cpu())
            pool_values.append(values.detach().cpu())
            if len(pool_values) > 10:
                break
            
        inds = torch.cat(pool_inds, dim=0)
        values = torch.cat(pool_values, dim=0)
        inds = torch.topk(values, k=10)[1]
        # inds = torch.topk(torch.cosine_similarity(pool_rep, img_rep, dim=1), k=5)[1]
        inds = [i.item() for i in inds]
        pool_ids = list(map(lambda x: pool_df['image'].values[x], inds))
        pool_ids = list(map(lambda x: x.split('/')[-1], pool_ids))
        print(pool_ids)
        assert len(pool_ids) == 10
        
        labels = fcp.request_labels(tm, tm_pw, pool_ids)
        
def get_all_data_with_labels(tm, tm_pw, id_dir, nw_dir):
    """
        Get the team data that has labels (initial data plus requested data).
        Writes a csv file with the image names and their class ids.
        Also writes a csv file with some useful statistics
    """
    stats_out = os.path.join(nw_dir, "fungi_class_stats.csv")
    data_out = os.path.join(nw_dir, "data_with_labels.csv")

    imgs_and_data = fcp.get_data_set(tm, tm_pw, 'train_labels_set')
    imgs_and_data_r = fcp.get_data_set(tm, tm_pw, 'requested_set')
    print("Team", tm, ' has access to images with labels:\n',
          'Basis set:', len(imgs_and_data), '\n',
          'Requested set:', len(imgs_and_data_r))

    total_img_data = imgs_and_data + imgs_and_data_r
    df = pd.DataFrame(total_img_data, columns=['image', 'taxonID'])
    # print(df.head())
    all_taxon_ids = df['taxonID']

    # convert taxonID into a class id
    taxon_id_to_label = {}
    # label_to_taxon_id = {}
    for count, value in enumerate(all_taxon_ids.unique()):
        taxon_id_to_label[int(value)] = count
        # label_to_taxon_id[count] = int(value)

    with open(data_out, 'w') as f:
        f.write('image,class\n')
        for t in total_img_data:
            class_id = taxon_id_to_label[t[1]]
            out_str = os.path.join(id_dir, t[0]) + '.JPG, ' + str(class_id) + '\n'
            f.write(out_str)

    with open(stats_out, 'w') as f:
        f.write('taxonID,class,count\n')
        for ti in taxon_id_to_label:
            count = df['taxonID'].value_counts()[ti]
            class_id = taxon_id_to_label[ti]
            out_str = str(ti) + ', ' + str(class_id) + ', ' + str(count) + '\n'
            f.write(out_str)     
     
        
if __name__ == '__main__':
    # Your team and team password
    # team = "DancingDeer"
    # team_pw = "fungi44"
    team = "SwimmingApe"
    team_pw = "fungi18"

    # where is the full set of images placed
    image_dir = "/scratch/hilss/DF20M/"

    # where should log files, temporary files and trained models be placed
    network_dir = "./network/"
    
    get_similar_images(network_dir, team, team_pw)
    

