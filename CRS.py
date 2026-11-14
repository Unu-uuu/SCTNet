import torch
import torch.nn.functional as F
from torch import nn


def Norm(x):
    device = x.device
    max_val_t = torch.max(x, 2)[0]
    max_val = torch.max(max_val_t, 2)[0]

    min_val_t = torch.min(x, 2)[0]
    min_val = torch.min(min_val_t, 2)[0]

    delta_t1 = torch.sub(max_val, min_val)
    delta_t2 = torch.unsqueeze(delta_t1, 2)
    delta = torch.unsqueeze(delta_t2, 3)

    min_val_t1 = torch.unsqueeze(min_val, 2)
    min_val = torch.unsqueeze(min_val_t1, 3)

    rel_t1 = torch.sub(x, min_val)
    rel_t2 = torch.div(rel_t1, delta)
    rel = torch.mul(rel_t2, 255).int()
    return rel.to(device)


def Entropy(x):
    device = x.device
    B, C, W, H = x.size()
    size = W * H
    histic = torch.zeros(size=(B, C, 256), device=device)
    for i in range(256):
        eq_i = torch.eq(x, i)
        sum_t1 = torch.sum(eq_i, dim=2)
        sum = torch.sum(sum_t1, dim=2)
        histic[:, :, i] = sum
    p_ij = torch.div(histic, size)
    h_ij_t1 = torch.add(p_ij, 1e-8)
    h_ij_t2 = p_ij * torch.log(h_ij_t1)
    h_ij = -torch.sum(h_ij_t2, dim=2)
    return torch.unsqueeze(torch.unsqueeze(h_ij, 2), 3).to(device)


def Cos_Similarity(x, y):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    result = cos(x, y)
    # 确保返回值有正确的维度，不使用mean
    # 如果结果是一维的，保持一维；如果是标量，保持标量
    return result


def Smooth(tensor, a=0.02):
    device = tensor.device
    temp_norm = torch.norm(tensor.float(), p=2, dim=1)
    map_1 = (temp_norm >= a).float()
    map_s = (temp_norm < a).float()
    comb = map_1 * temp_norm + map_s * (0.5 * torch.pow(temp_norm, 2)/a + a * 0.5)
    return comb.to(device)


def Complementary_Learning_Loss(ms_v, pan_v):
    # 确保输入在正确的设备上
    device = ms_v.device
    
    # 计算均值，保持batch维度
    ms_v = torch.mean(ms_v, dim=1, keepdim=True)
    pan_v = torch.mean(pan_v, dim=1, keepdim=True)

    # 特征的余弦相似度
    cos_similarity = Cos_Similarity(ms_v, pan_v)

    # 特征差异性
    ms_v = Norm(ms_v).to(device)
    pan_v = Norm(pan_v).to(device)
    ms_v_imp = Entropy(ms_v).to(device)
    pan_v_imp = Entropy(pan_v).to(device)

    x = torch.sub(ms_v_imp, pan_v_imp)
    k = torch.ones_like(cos_similarity).to(device) / (torch.pow(cos_similarity, 2) + 0.001)

    # 惩罚权重
    w1 = torch.sigmoid(k * x)
    w2 = 1 - w1

    # 确保Smooth函数返回正确的维度
    smooth_ms_v = Smooth(ms_v)
    smooth_pan_v = Smooth(pan_v)
    
    # 确保所有张量在正确的设备上并且维度匹配
    w1 = w1.to(device)
    w2 = w2.to(device)
    
    # 确保维度匹配：w1和smooth_ms_v应该可以相乘
    # 如果w1是标量，直接相乘；如果w1是tensor，需要调整维度
    if w1.dim() > 0 and smooth_ms_v.dim() > 0:
        # 调整维度以匹配
        while w1.dim() < smooth_ms_v.dim():
            w1 = w1.unsqueeze(-1)
        while w2.dim() < smooth_pan_v.dim():
            w2 = w2.unsqueeze(-1)
    
    loss = w1 * smooth_ms_v + w2 * smooth_pan_v

    # 归一化损失到合理范围（0-10）
    # 使用除法缩放，因为当前值在140左右
    loss = loss / 1000.0  # 除以100，将140降到1.4

    # 将损失转换为标量，确保所有维度都被平均掉
    # 使用多个mean确保返回真正的标量
    while loss.dim() > 0:
        loss = loss.mean()
    
    return loss
