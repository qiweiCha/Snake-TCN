import argparse
import logging
import os
import sys
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import cv2

from lib.vss_net_SDK_TCN import VSS_Net_SDK_TCN
from torch.utils.data import DataLoader, random_split
from utils.dataloader import get_loader
from utils.eval import to_one_hot,AverageMeter,MetricsTracker,get_metrics,recompone_overlap,get_color,count_connect_component
from utils.helpers import *
from PIL import Image

def get_labels(labels_path):
    files = list(sorted(os.listdir(labels_path)))
    label_list = []
    for label in files:
        gt = cv2.imread(os.path.join(labels_path, label), cv2.IMREAD_GRAYSCALE)
        gt = np.where(gt >= 100, 1.0, 0.0)
        label_list.append(gt)  
    return label_list
       
def eval_net(net, loader, save_path,labels_path,device,size,stride):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)
    pres=[]
    img_savePath=os.path.join(save_path,'img_results')
    dir_exists(img_savePath)
    gts=get_labels(labels_path)
    metrics_tracker = MetricsTracker()
    with tqdm(total=n_val, desc='测试推理中', unit='batch', leave=False) as pbar:
        with torch.no_grad(): 
            for batch in loader:
                imgs = batch
                imgs = imgs.to(device=device, dtype=torch.float32)
                with torch.amp.autocast('cuda', enabled=True):
                    pre= net(imgs)

                pre = torch.softmax(pre, dim=1)[:,1,:,:]
                pres.extend(pre)

                pbar.update()

    remove_files(img_savePath)
    pres = torch.stack(pres, 0).cpu()

    H,W = gts[0].shape
    num_data = len(gts)
    pad_h = stride - (H - size[0]) % stride
    pad_w = stride - (W - size[1]) % stride
    new_h = H + pad_h
    new_w = W + pad_w
    pres = recompone_overlap(np.expand_dims(pres.cpu().detach().numpy(),axis=1), new_h, new_w, stride, stride)  # predictions
    predict = pres[:,0,0:H,0:W]
    predict_b = np.where(predict >= 0.5, 1, 0)

    for j in tqdm(range(num_data), desc="保存进度"):
            # cv2.imwrite(img_savePath + f"/gt{j}.png", np.uint8(gts[j]*255))
            cv2.imwrite(img_savePath + f"/pre{j}.png", np.uint8(predict[j]*255))
            # cv2.imwrite(img_savePath + f"/pre_b{j}.png", np.uint8(predict_b[j]*255))
            cv2.imwrite(img_savePath + f"/color_b{j}.png", get_color(predict_b[j],gts[j]))
            metrics=get_metrics(predict[j], gts[j])
            metrics_tracker.update_metrics(*metrics)
    
    
    mean_metrics = metrics_tracker.get_metrics_mean()
    logger.info(f'## test  means results ## ')
    for k, v in mean_metrics.items():
        logger.info(f'{str(k):15s}: {v}')
    save_metrics_to_file(mean_metrics,save_path)

def test_net(net,
              device,
              save_path,
              batch_size=64,
              stride=32,
              img_size=(64,64)):

    
    images_path = ''
    labels_path = ''
    test_loader = get_loader(images_path, labels_path, batch_size=batch_size, size=img_size, shuffle=False,is_train=False,stride=stride)
    net.eval()

    eval_net(net, test_loader,save_path,labels_path, device,size=img_size,stride=stride)


def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=64,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-f', '--load', dest='load', type=str, default='',
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--img_size', dest='size', type=int, default=(64,64),
                        help='The size of the images')
    parser.add_argument('-r', '--stride',dest='stride' ,type=int, default=32,
                        help='The stride ')

    return parser.parse_args()


if __name__ == '__main__':
   
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    torch.cuda.empty_cache() 
    net = VSS_Net_SDK_TCN()
    net = nn.DataParallel(net, device_ids=[0])
    net.to(device=device)

    if args.load:
        
        checkpoint = torch.load(args.load, map_location=device)  

        net.load_state_dict(checkpoint['model_state_dict'], strict=False)  

        logging.info(f'Model loaded from {args.load}')
    
    try:
        test_net(net=net,
                  batch_size=args.batchsize,
                  device=device,
                  img_size=args.size,
                  stride=args.stride,
                  save_path=args.load.rsplit('/',1)[0])
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
