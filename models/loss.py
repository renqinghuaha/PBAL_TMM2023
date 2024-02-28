import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelContrastLoss(nn.Module):
    def __init__(self, cfg):
        super(PixelContrastLoss, self).__init__()
        self.cfg = cfg
        self.temperature = 0.07

    def contrast_cos_calc(self, x1, x2):
        n, c, h, w = x1.shape
        X_ = F.normalize(x1, p=2, dim=1)
        X_ = X_.permute(0, 2, 3, 1).contiguous().view(-1, c)
        x2 = x2.contiguous().view(-1, c)
        Y_ = F.normalize(x2, p=2, dim=-1)
        out = torch.matmul(X_, Y_.T)
        out = out.contiguous().view(n, h, w, 19).permute(0, 3, 1, 2)
        return out

    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, 20, w, h).cuda()
        id = torch.where(label < 19, label, torch.Tensor([19]).cuda())
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1[:, :19, :, :]

    def _contrastive(self, label, contrast):
        n, c, h, w = contrast.size()
        label = self.process_label(label)
        anchor_dot_contrast = contrast.permute(0, 2, 3, 1).contiguous().view(-1, c)
        anchor_dot_contrast = torch.div(anchor_dot_contrast, self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = label.permute(0, 2, 3, 1).contiguous().view(-1, c)
        neg_mask = 1 - mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)
        exp_logits = torch.exp(logits)
        log_prob = torch.log(exp_logits + neg_logits) - logits
        loss = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        loss = loss.mean()
        return loss

    def forward(self, feats, labels, queue):
        mask = F.interpolate(labels.unsqueeze(1).float(), size=feats.shape[-2:], mode='nearest')
        feat_contrast = self.contrast_cos_calc(feats, queue)
        loss = self._contrastive(mask, feat_contrast)
        return loss