import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
import numpy as np
import cv2
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from utils.cldice import clDice
# from utils.cldiceC import clDice
def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape) == 4)
    assert (preds.shape[1] == 1 or preds.shape[1] == 3)
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_img = N_patches_h * N_patches_w
    assert (preds.shape[0] % N_patches_img == 0)
    N_full_imgs = preds.shape[0] // N_patches_img
    full_prob = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))
    full_sum = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))
    k = 0
    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                full_prob[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += preds[
                    k]
                full_sum[i, :, h * stride_h:(h * stride_h) + patch_h,
                         w * stride_w:(w * stride_w) + patch_w] += 1
                k += 1
    assert (k == preds.shape[0])
    assert (np.min(full_sum) >= 1.0)
    final_avg = full_prob / full_sum
    return final_avg

def get_color(predict, target):
    """
    绘制预测结果对比图:
    - True Positive (TP) -> Green (绿色)
    - False Positive (FP) -> Red (红色)
    - False Negative (FN) -> Blue (蓝色)
    - True Negative (TN) -> Black (黑色, 背景)
    """
    H, W = predict.shape
    
    # 初始化一个全黑的图像 (默认作为 TN: True Negative)
    # 使用 uint8 类型是图像处理的标准做法
    img_colour = np.zeros((H, W, 3), dtype=np.uint8)

    # 定义颜色 (R, G, B)
    COLOR_TP = [0, 255, 0]    # 绿色
    COLOR_FP = [0, 0, 255]    # 红色
    COLOR_FN = [255, 0, 0]    # 蓝色

    # --- 使用 NumPy 布尔掩码进行赋值 (比 for 循环快几百倍) ---

    # 1. True Positive (TP): 预测为1 且 真实为1
    mask_tp = (predict == 1) & (target == 1)
    img_colour[mask_tp] = COLOR_TP

    # 2. False Positive (FP): 预测为1 但 真实为0
    mask_fp = (predict == 1) & (target == 0)
    img_colour[mask_fp] = COLOR_FP

    # 3. False Negative (FN): 预测为0 但 真实为1
    mask_fn = (predict == 0) & (target == 1)
    img_colour[mask_fn] = COLOR_FN

    # 4. True Negative (TN): 剩余部分保持初始化时的黑色 [0, 0, 0]

    return img_colour

def count_connect_component(predict, target, connectivity=8):

    pre_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        predict, dtype=np.uint8)*255, connectivity=connectivity)
    gt_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        target, dtype=np.uint8)*255, connectivity=connectivity)
    return pre_n/gt_n

def to_one_hot(seg, all_seg_labels=None):
    if all_seg_labels is None:
        all_seg_labels = np.unique(seg)
    result = np.zeros((len(all_seg_labels), *seg.shape), dtype=seg.dtype)
    for i, l in enumerate(all_seg_labels):
        result[i][seg == l] = 1
    return result

class AverageMeter(object):
    def __init__(self):
        self.val = []

    def update(self, val):
        self.val.append(val)

    @property
    def mean(self):
        return np.round(np.mean(self.val), 4)

    @property
    def std(self):
        return np.round(np.std(self.val), 4)

class MetricsTracker:
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        # 重置监控指标
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.loss = AverageMeter()
        self.auc = AverageMeter()
        self.DSC = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.iou = AverageMeter()
        self.VC = AverageMeter()
        self.cldice = AverageMeter()
        self.pre = AverageMeter()

    def update_metrics(self, DSC , iou , cldice, acc, sen, spe, auc,pre):
        # 更新监控指标
        self.DSC.update(DSC)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.iou.update(iou)
        self.auc.update(auc)
        self.cldice.update(cldice)
        self.pre.update(pre)

    def get_metrics_mean(self):
        # 返回监控指标的均值
        return {
            "DSC": self.DSC.mean,
            "Acc": self.acc.mean,
            "Sen": self.sen.mean,
            "Spe": self.spe.mean,
            "IOU": self.iou.mean,
            "AUC": self.auc.mean,
            "cldice": self.cldice.mean,
            "Pre": self.pre.mean,
        }

def get_metrics(predict, target, run_clDice=True, threshold=0.5):
    predict_b = np.where(predict >= threshold, 1, 0)
    
    cldice = clDice(predict_b, target) if run_clDice else 0
    predict = predict.flatten()
    predict_b = predict_b.flatten()
    target = target.flatten()
    
    if max(target) > 1:
        target = to_one_hot(target, all_seg_labels=[1]).flatten()
    
    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()
    
    if np.all(target == 0) or np.all(predict == 0):
        auc = 1
    else:
        auc = roc_auc_score(target, predict)

    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp)!=0)
    sen = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn)!=0)
    spe = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp)!=0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fp + fn)!=0)
    
    
    # 使用 np.divide 安全处理 DSC 的计算
    numerator = 2 * pre * sen
    denominator = pre + sen
    DSC = np.divide(numerator, denominator, 
                   out=np.zeros_like(numerator, dtype=float), 
                   where=denominator!=0)
    
    return DSC , iou , cldice, acc, sen, spe, auc,pre

def eval_net(net, loader, device, batch_size):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    metrics_tracker = MetricsTracker()
    mean_metrics = {}
    n_val = len(loader)
    DC_loss = DC_and_CE_loss({}, {})
    total_loss = 0.0  # 使用浮点数
    batch_num=0
    with tqdm(total=n_val, desc='【Val】', unit='batch') as pbar:
        for _ , batch in enumerate(loader):
            imgs, gts = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            gts = gts.to(device=device, dtype=torch.long)

            with torch.no_grad():
                preds= net(imgs)  # B,1,H,W
                loss = DC_loss(preds, gts)
            total_loss += loss.item()  
            batch_num+=1
            metrics = get_metrics(torch.softmax(preds, dim=1).cpu().detach().numpy()[:, 1, :, :], gts.squeeze(1).cpu().detach().numpy())            
            metrics_tracker.update_metrics(*metrics)
            
            pbar.set_description(
                f"Loss: {loss.item():.4f} | "
                f"DSC: {metrics[0]:.4f} | "
                f"IOU: {metrics[1]:.4f} | "
                f"AUC: {metrics[6]:.4f} | "
            )
            pbar.update()
    epoch_val_loss = total_loss / n_val  # 批次平均      
    mean_metrics = metrics_tracker.get_metrics_mean()
    logger.info(f'## val  means results ## ')
    for k, v in mean_metrics.items():
        logger.info(f'{str(k):15s}: {v}')
    
    return mean_metrics['DSC'], epoch_val_loss