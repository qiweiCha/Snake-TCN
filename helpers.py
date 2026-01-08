import os
import numpy as np
import torch.nn.functional as F
from typing import List, Union
import torch
import cv2
import csv
import pandas as pd
import torch
from torchvision.utils import make_grid
from batchgenerators.utilities.file_and_folder_operations import *
from datetime import datetime
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

def visualize_and_save(mask, pred_dir, gt_dir, step=4, scale=20, save_path=None):
    """在血管mask上同时画预测方向和GT方向"""
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if torch.is_tensor(pred_dir):
        pred_dir = pred_dir.cpu().numpy()
    if torch.is_tensor(gt_dir):
        gt_dir = gt_dir.cpu().numpy()

    H, W = mask.shape
    Y, X = np.mgrid[0:H:step, 0:W:step]

    # 采样预测方向
    U_pred = pred_dir[0, ::step, ::step]
    V_pred = pred_dir[1, ::step, ::step]

    # 采样GT方向
    U_gt = gt_dir[0, ::step, ::step]
    V_gt = gt_dir[1, ::step, ::step]

    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap="gray")
    plt.quiver(X, Y, U_gt, V_gt, color="blue", scale=scale, headwidth=3, headlength=5, label="GT")
    plt.quiver(X, Y, U_pred, V_pred, color="yellow", scale=scale, headwidth=3, headlength=5, label="Pred")
    plt.legend()
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compute_gt_direction(mask):
    """
    根据二值血管 mask 生成方向 GT
    Args:
        mask: [H, W] 0/1 二值血管标签 (numpy array 或 torch tensor)
    Returns:
        gt_dir: [2, H, W] 归一化方向场 (dx, dy)
    """
    # 转 numpy 并保证是 0/1
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    mask = (mask > 0).astype(np.uint8)

    # 1. 中心线提取
    skeleton = skeletonize(mask).astype(np.uint8)

    # 2. Sobel 计算梯度方向
    dx = cv2.Sobel(skeleton, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(skeleton, cv2.CV_64F, 0, 1, ksize=3)

    # 3. 避免 0 向量
    mag = np.sqrt(dx**2 + dy**2) + 1e-8
    dx /= mag
    dy /= mag

    # 4. 拼成 [2, H, W] 向量场
    gt_dir = np.stack([dx, dy], axis=0)

    return torch.tensor(gt_dir, dtype=torch.float32)

def remove_files(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

def save_metrics_to_file(metrics_dict, save_path):
    """
    将测试指标保存到文本文件
    
    Args:
        metrics_dict (dict): 指标字典 {指标名: 值}
        file_path (str): 保存路径，默认当前目录的test_result.txt
    """
    file_path = os.path.join(save_path, "test_result.txt")
    try:
        with open(file_path, 'a') as f:
            f.write("===== 模型评估结果 =====\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 写入指标（按15字符对齐）
            max_len = max(len(str(k)) for k in metrics_dict.keys())
            for k, v in metrics_dict.items():
                f.write(f"{str(k).ljust(max_len)} : {v}\n")
                
            print(f"✅ 评估结果已保存至: {os.path.abspath(file_path)}")
            
    except Exception as e:
        print(f"❌ 保存失败: {str(e)}")

def save_metrics_and_plot(
    val_dsc: float,         # 验证集 DSC (Dice Score)
    val_loss: float,        # 验证集损失 (Validation Loss)
    epoch: int,             # 当前 epoch 索引 (从 0 开始)
    epoch_loss: float,      # 训练集损失 (Train Loss)
    save_dir: str
):
    """
    将训练和验证指标记录到 CSV 文件，并生成损失和 DSC 的变化曲线图。

    Args:
        val_dsc (float): 当前 epoch 的验证集 Dice Score (DSC)。
        val_loss (float): 当前 epoch 的验证损失。
        epoch (int): 当前的 epoch 索引 (从 0 开始)。
        epoch_loss (float): 当前 epoch 的训练损失。
        save_dir (str): 保存文件和图表的目录路径。
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 转换 Tensor 为标量数字 ---
    def get_scalar_value(value):
        """
        如果输入值是带有 .item() 方法的对象 (如 PyTorch Tensor)，
        则使用 .item() 提取其标量值，否则返回原值。
        """
        # 检查对象是否具有 .item() 方法，这适用于 PyTorch/TensorFlow 标量
        if hasattr(value, 'item'):
            return value.item()
        return value

    final_epoch_loss = get_scalar_value(epoch_loss)
    final_val_loss = get_scalar_value(val_loss)
    final_val_dsc = get_scalar_value(val_dsc) # DSC 也可能来自 tensor

    # --- 1. 记录损失历史到 training_history.csv ---
    history_file = os.path.join(save_dir, 'training_history.csv')
    is_new_file = not os.path.exists(history_file)
    
    # 写入当前数据到 CSV
    with open(history_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 如果是新文件，写入表头
        if is_new_file:
            # 新增 'Val_AUC' 列
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_AUC'])
            
        # 写入当前数据
        # epoch + 1 是为了显示人类可读的 epoch 编号
        writer.writerow([epoch + 1, final_epoch_loss, final_val_loss, final_val_dsc])
    
    print(f"指标历史已记录到: {history_file}")

    # --- 2. 读取历史记录并生成图表 (需要将读取操作放在写入操作的 with 块之外) ---
    try:
        # 使用 pandas 读取完整的历史数据
        history_df = pd.read_csv(history_file)

        # 设置绘图风格
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # --- 2.1 生成损失曲线图 (loss_curve.png) ---
        plt.figure(figsize=(20, 10))
        
        plt.plot(history_df['Epoch'], history_df['Train_Loss'], 
                 label='Train Loss', 
                 marker='o', linestyle='-', linewidth=2, color='#1f77b4')
        
        plt.plot(history_df['Epoch'], history_df['Val_Loss'], 
                 label='Val Loss', 
                 marker='s', linestyle='--', linewidth=2, color='#ff7f0e')
        
        plt.title('loss_curve', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.xticks(history_df['Epoch'].astype(int))
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plot_path_loss = os.path.join(save_dir, 'loss_curve.png')
        plt.savefig(plot_path_loss, dpi=300, bbox_inches='tight')
        plt.close() # 关闭图形，释放内存
        print(f"损失曲线图已保存到: {plot_path_loss}")

        # --- 2.2 生成 DSC 曲线图 (dsc_curve.png) ---
        plt.figure(figsize=(10, 6))
        
        # 绘制 DSC (Dice Score) 曲线
        plt.plot(history_df['Epoch'], history_df['Val_DSC'], 
                 label='val_DSC', 
                 marker='^', linestyle='-', linewidth=2, color='green') # 使用不同的颜色和标记
        
        plt.title(' DSC_curve', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Dice Score', fontsize=14)
        # 注意：对于 Dice Score (指标)，通常希望它上升，可以添加 y 轴范围 0-1
        plt.ylim(0, 1) 
        plt.legend(fontsize=12)
        plt.xticks(history_df['Epoch'].astype(int))
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plot_path_dsc = os.path.join(save_dir, 'dsc_curve.png')
        plt.savefig(plot_path_dsc, dpi=300, bbox_inches='tight')
        plt.close() # 关闭图形，释放内存
        print(f"DSC 曲线图已保存到: {plot_path_dsc}")


    except Exception as e:
        print(f"生成图表时发生错误: {e}")

def create_temporal_grid(sequence, save_dir):
    """将时间序列帧拼接成网格图"""
    # sequence: (T,H,W)
    frames = [torch.from_numpy(frame) for frame in sequence]
    grid = make_grid(frames, nrow=5, padding=10)  # 每行4帧
    grid = grid.numpy().transpose(1, 2, 0)  # (H,W,C)
    cv2.imwrite(os.path.join(save_dir, "temporal_grid.png"), grid[:,:,0])  # 单通道

def visualize_batch(imgs, gt, save_dir="visualization"):
    """
    Args:
        imgs: (B,T,H,W) 输入图像序列
        gt: (B,1,H,W) 标签
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 转换为numpy并调整值范围
    imgs_np = imgs.cpu().numpy()  # (B,T,H,W)
    gt_np = gt.cpu().numpy()[:, 0]  # (B,H,W) 去掉通道维度
    
    # 反归一化（如果之前做过归一化）
    # 需要逐帧处理
    restored = np.empty_like(imgs_np, dtype=np.uint8)
    for b in range(imgs_np.shape[0]):
        for t in range(imgs_np.shape[1]):
            restored[b,t] = (imgs_np[b,t] * 255).clip(0, 255)
    imgs_np = restored
    gt_np = (gt_np * 255).clip(0, 255).astype(np.uint8)
    
    for batch_idx in range(imgs.shape[0]):
        # 为当前样本创建子目录
        sample_dir = os.path.join(save_dir, f"sample_{batch_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 保存时间序列图像
        for t in range(imgs.shape[1]):
            frame = imgs_np[batch_idx, t]
            cv2.imwrite(os.path.join(sample_dir, f"frame_{t:03d}.png"), frame)
        
        # 保存GT
        cv2.imwrite(os.path.join(sample_dir, "gt.png"), gt_np[batch_idx])
        
        # 创建并保存时间序列的网格图（可选）
        create_temporal_grid(imgs_np[batch_idx], sample_dir)

def to_patch(
    img_list: List[Union[torch.Tensor, np.ndarray]],
    size: tuple = (256, 256),
    stride: int = None,
    pad_value: float = 0.0
) -> List[torch.Tensor]:
    """
    将图片列表分块为指定大小的patch
    
    Args:
        img_list: 输入列表，每个元素为(C,H,W)的tensor或ndarray
        size: 目标patch大小 (h, w)
        stride: 滑动步长（默认等于size即不重叠）
        pad_value: 填充值
        
    Returns:
        List[torch.Tensor]: 分块后的patch列表，每个元素为(C,h,w)
    """
    # 参数校验
    assert len(size) == 2, "Patch size must be (height, width)"
    if stride is None:
        stride = size  # 默认不重叠
    
    patches = []
    for img in img_list:
        # 统一转为torch tensor
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        
        C, H, W = img.shape
        h, w = size
        
        # 计算需要填充的像素数 (右和下)
        pad_h = (h - H % h) % h
        pad_w = (w - W % w) % w
        
        # 对称填充 (方便边缘信息保留)
        img_padded = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=pad_value)
        
        # 使用unfold分块 (高效实现)
        patches.append(
            img_padded.unfold(1, h, stride[0])  # 沿高度滑动
               .unfold(2, w, stride[1])         # 沿宽度滑动
               .reshape(C, -1, h, w)            # 合并分块维度
               .permute(1, 0, 2, 3)             # (N_patches, C, h, w)
        )
    
    # 合并所有图片的patch
    return torch.cat(patches, dim=0)
