import argparse
import logging
import os
import sys
import math # <--- MODIFIED: 引入 math 库以使用 -inf

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from timm.scheduler.cosine_lr import CosineLRScheduler
from utils.eval import eval_net
from lib.Snake_TCN import Snake_TCN
from datetime import datetime
from utils.dataloader import get_loader
from utils.losses import *
from utils.eval import MetricsTracker,get_metrics
from utils.helpers import *


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def cal(loader):
    tot = 0
    for batch in loader:
        imgs, _ = batch
        tot += imgs.shape[0]
    return tot


def train_net(net,
              device,
              epochs,
              batch_size,
              img_size,
              lr=5e-4,
              save_cp=True,
              patience=15 # 早停的patience参数
              ):

    train_loader = get_loader(train_img_dir, train_mask_dir, batch_size=batch_size,shuffle=True ,size=img_size,stride=int(img_size[0]/2),is_train=True)
    val_loader = get_loader(val_img_dir, val_mask_dir, batch_size=batch_size,shuffle=False ,size=img_size,stride=int(img_size[0]/2),is_train=False)
    n_train = cal(train_loader)
    n_val = cal(val_loader)
    num_steps = len(train_loader) 
    logger = get_logger('DSCA.log')
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    
    save_dir = os.path.join(save_path, model_name + '_' + current_time)

    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError:
        pass


    optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=lr,                  
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.05,
            )
    
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=epochs * num_steps, 
        cycle_mul=1.,
        lr_min=5e-6,                  
        warmup_t=10 * num_steps,      
        warmup_lr_init=5e-7,          
        cycle_limit=1,
        t_in_epochs=False,            
    )

    DC_loss = DC_and_CE_loss({},{})

    scaler = torch.amp.GradScaler(enabled=True)

    
    best_dsc = -math.inf 
    epochs_no_improve = 0
    start_epoch = 0

    checkpoint_path = os.path.join(save_dir, 'best_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        logger.info(f'Found checkpoint at {checkpoint_path}. Attempting to load...')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        
        best_dsc = checkpoint.get('best_dsc', -math.inf) 
        epochs_no_improve = checkpoint['epochs_no_improve']
        
        logger.info('--- Checkpoint loaded successfully! ---')
        logger.info(f'Resuming training from Epoch {start_epoch}.')
        logger.info(f'Current best_dsc: {best_dsc:.4f}')
        logger.info(f'Current epochs_no_improve: {epochs_no_improve}')

    logger.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Vailding size:   {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {img_size}
        Patience:        {patience}
        Start Epoch:     {start_epoch}
    ''')

    for epoch in range(start_epoch, epochs):
        net.train()
        metrics_tracker = MetricsTracker()
        epoch_loss = 0.0 
        b_cp = False
        with tqdm(total=n_train, desc=f'【Training】Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                imgs, gt = batch
                imgs = imgs.to(device=device, dtype=torch.float32)
                gt = gt.to(device=device, dtype=torch.long)

                with torch.amp.autocast('cuda', enabled=True):     
                    pre= net(imgs)
                    loss = DC_loss(pre, gt)
                
                scaler.scale(loss).backward()
                
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                lr_scheduler.step_update(epoch * num_steps + batch_idx)
                
                epoch_loss += loss.item()

                with torch.no_grad():
                    probs = torch.softmax(pre, dim=1)[:, 1, :, :].cpu().numpy()
                    gt_np = gt.squeeze(1).cpu().numpy()
                    metrics = get_metrics(probs, gt_np)
                    metrics_tracker.update_metrics(*metrics)

                pbar.set_description(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.8f} | "
                    f"DSC: {metrics[0]:.4f} | "
                    f"AUC: {metrics[6]:.4f} | "
                )
                pbar.update(imgs.shape[0])
        
        epoch_loss /= len(train_loader)
        mean_metrics = metrics_tracker.get_metrics_mean()

        
        val_dsc,val_loss= eval_net(net, val_loader, device, batch_size)

        lr_scheduler.step(val_loss)

        if val_dsc >= best_dsc:
           best_dsc = val_dsc
           epochs_no_improve = 0
           b_cp = True
        else:
           epochs_no_improve += 1
           b_cp = False
           logging.info(f'EarlyStopping counter: {epochs_no_improve} out of {patience}')

        if epochs_no_improve >= patience:
            logging.info(f'Early stopping triggered after {patience} epochs without improvement.')
            break

        if save_cp and b_cp:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_dsc': best_dsc, # 保存AUC
                'epochs_no_improve': epochs_no_improve,
                'img_size': img_size,
                'batch_size': batch_size,
                'lr': lr,
            }
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Best checkpoint saved!|epoch:{epoch + 1}, val_AUC:{val_dsc:.4f}")

        save_metrics_and_plot(val_dsc, val_loss, epoch, epoch_loss, save_dir)
         

def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=128,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=5e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-s', '--img_size', dest='size', type=int, default=(64,64),
                        help='The size of the images')
    parser.add_argument('--optimizer', type=str,
                        default='adamw', help='choosing optimizer Adam or SGD')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=30, help='every n epochs decay learning rate')
   
    return parser.parse_args()

root_path=""
train_img_dir = ''
train_mask_dir = ''
val_img_dir = ''
val_mask_dir = ''
save_path = 'savepth/'
model_name=''

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = snake_tcn()
    model_name='snake_tcn'

    net = nn.DataParallel(net, device_ids=[0])
    net = net.to(device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_size=args.size)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
