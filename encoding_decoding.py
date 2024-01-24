import torch
from torch import Tensor
from torch.nn import functional as F
import config



def unbiased_encoding(loc:Tensor, reso:int, angle_range:int=config.angle_range):
    """
    loc: location of a speaker
    reso: resolution, e.g. the number of cells ($I$ of this paper)
    angle_range: the maximum range of DOA, is 180 or 360
    """
    cell_len = angle_range / reso
    num_classes = reso + 1
    loc[loc==config.angle_range] -= 1e-4    # Prevent crossing boundaries
    class0 = torch.div(loc, cell_len, rounding_mode='trunc').long()
    class1 = class0 + 1
    w1 = torch.fmod(loc, cell_len) / cell_len
    w1 = w1.view(-1, 1)  # code.shape = (-1, num_classes), w.shape = (-1, 1)
    w0 = 1 - w1
    code = F.one_hot(class0, num_classes)*w0 + \
           F.one_hot(class1, num_classes)*w1
    return code



def weighted_adjacent_decoding(logits:Tensor, selected_classes:int):
    """
    logits: Predicted distribution
    selected_classes: Number of classes selected for weighted adjacent decoding
    """
    num_classes = logits.shape[-1]
    k = logits.argmax(dim=-1, keepdim=True)

    k_l = k - 1
    k_l[k_l<0] += 2     # There is only one adjacent grid, making the left grid equal to the right grid
    k_r = k + 1
    k_r[k_r==num_classes] -= 2  # There is only one adjacent grid, making the left grid equal to the right grid
    logit_c = torch.gather(logits, -1, k) # shape = (batch_size, 1)
    logit_l = torch.gather(logits, -1, k_l)
    logit_r = torch.gather(logits, -1, k_r)
    mask_unique = torch.ne(logit_l, logit_r)
    logit_r = logit_r * mask_unique     # Handling boundary situations

    if selected_classes==1:
        result = k.float()
    
    elif selected_classes==2:
        k_h = torch.where(logit_l > logit_r, k_l, k_r)
        logit_h = torch.gather(logits, -1, k_h)
        sum_prob = logit_c + logit_h
        result = (logit_c*k + logit_h*k_h) / sum_prob

    elif selected_classes==3:
        sum_prob = logit_c + logit_l + logit_r
        result = (logit_c*k + logit_l*k_l + logit_r*k_r) / sum_prob
    
    return result.view(result.shape[:-1])    # shape = (batch_size)



def onehot_encoding(loc:Tensor, reso:int, angle_range:int=config.angle_range):
    """
    loc: location of a speaker
    reso: resolution, e.g. the number of cells ($I$ of this paper)
    angle_range: the maximum range of DOA, is 180 or 360
    """
    cell_len = angle_range / reso
    num_classes = reso + 1
    gt = torch.round(loc / cell_len).long()
    code = F.one_hot(gt, num_classes=num_classes)
    return code.float()



def gaussian_encoding(sigma, loc:Tensor, reso:int, angle_range:int=config.angle_range):
    """
    sigma: standard deviation
    loc: location of a speaker
    reso: resolution, e.g. the number of cells ($I$ of this paper)
    angle_range: the maximum range of DOA, is 180 or 360
    """
    cell_len = angle_range / reso
    num_classes = reso + 1
    gt = loc.view(-1, 1)
    idx = torch.arange(num_classes).unsqueeze(0) * cell_len
    code = torch.exp(-((idx - gt)**2 / (sigma**2)))
    return code



def soft_encoding(loc:Tensor, reso:int, angle_range:int=config.angle_range):
    """
    loc: location of a speaker
    reso: resolution, e.g. the number of cells ($I$ of this paper)
    angle_range: the maximum range of DOA, is 180 or 360
    """
    cell_len = angle_range / reso
    num_classes = reso + 1
    gt = torch.round(loc.view(-1, 1) / cell_len).long()

    gt_2 = gt - 2
    gt_2[gt_2 < 0] += 4
    gt_1 = gt - 1
    gt_1[gt_1 < 0] += 2
    gt1 = gt + 1
    gt1[gt1 > reso] -= 2
    gt2 = gt + 2
    gt2[gt2 > reso] -= 4

    mask2 = torch.eq(gt_2, gt2).unsqueeze(1)
    mask1 = torch.eq(gt_1, gt1).unsqueeze(1)

    code_2 = F.one_hot(gt_2, num_classes) * 0.1
    code_1 = F.one_hot(gt_1, num_classes) * 0.2
    code_0 = F.one_hot(gt, num_classes) * 0.4
    code1 = F.one_hot(gt1, num_classes) * 0.2
    code2 = F.one_hot(gt2, num_classes) * 0.1

    code = code_2 + code_1 + code_0 + code1 + code2 - mask1*code1 - mask2*code_2
    return code
