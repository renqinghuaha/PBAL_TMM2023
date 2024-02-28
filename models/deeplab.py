import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.aspp import build_aspp
from models.decoder import build_decoder
from models.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet101', output_stride=16, num_classes=21,
                    bn='bn', freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        self.best_iou = 0
        if bn == 'sync_bn':
            BatchNorm = SynchronizedBatchNorm2d
        # elif bn == 'sync_abn':
        #     BatchNorm = InPlaceABNSync
        elif bn == 'bn':
            BatchNorm = nn.BatchNorm2d
        # elif bn == 'abn':
        #     BatchNorm = InPlaceABN
        elif bn == 'gn':
            BatchNorm = nn.GroupNorm
        else:
            raise NotImplementedError('batch norm choice {} is not implemented'.format(bn))

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.backbone._load_pretrained_model()
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def _load_pretrained_model(self):
        if hasattr(self.backbone, '_load_pretrained_model'):
            self.backbone._load_pretrained_model()

    def forward(self, input):
        out_dict = {}
        x, low_level_feat = self.backbone(input)
        # out_dict['out_aux'] = self.multi_classifier(x)
        x = self.aspp(x)
        cls_feat, out = self.decoder(x, low_level_feat)
        out_dict['feat'] = cls_feat
        out_dict['out'] = out
        return out_dict

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()
            elif isinstance(m, nn.GroupNorm):
                m.eval()
            elif m.__class__.__name__.find('BatchNorm')!= -1:
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


