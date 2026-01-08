import numpy as np
from skimage.morphology import skeletonize

def cl_score(v, s, epsilon=1e-7):
    """计算骨架体积重叠 (Skeleton Volume Overlap)"""
    sum_s = np.sum(s)
    return np.sum(v * s) / (sum_s + epsilon) if sum_s > 0 else 0.0

def clDice(v_p, v_l, epsilon=1e-7):
    """
    计算 clDice 指标
    Args:
        v_p: 预测二值图 (B,H,W) 或 (H,W), 值需为0/1
        v_l: 真实标签二值图 (B,H,W) 或 (H,W), 值需为0/1
    Returns:
        平均 clDice (标量)
    """
    # 输入验证
    v_p, v_l = np.asarray(v_p), np.asarray(v_l)
    assert v_p.shape == v_l.shape, "Shapes mismatch"
    assert np.all(np.isin(v_p, [0, 1])) and np.all(np.isin(v_l, [0, 1])), "Inputs must be binary"
    
    # 处理单图情况 (H,W) -> (1,H,W)
    if v_p.ndim == 2:
        v_p, v_l = v_p[np.newaxis], v_l[np.newaxis]
    
    # 预处理骨架（对标签只需计算一次）
    s_l = np.stack([skeletonize(mask) for mask in v_l])  # (B,H,W)
    s_p = np.stack([skeletonize(mask) for mask in v_p])  # (B,H,W)
    
    # 批量计算
    tprec = np.array([cl_score(v_p[i], s_l[i]) for i in range(len(v_p))])
    tsens = np.array([cl_score(v_l[i], s_p[i]) for i in range(len(v_p))])
    
    # 计算 clDice
    clDice = 2 * tprec * tsens / (tprec + tsens + epsilon)
    return np.mean(clDice)
