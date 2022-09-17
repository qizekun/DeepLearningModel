import torch
import torch.nn as nn
from module.stoch_norm import StochBatchNorm2d, StochNorm2d
from torchvision import models
from timm.models import vit_base_patch16_224_in21k

__all__ = ['VisionTransformer_F', 'VisionTransformer']


class VisionTransformer_F(nn.Module):
    def __init__(self, pretrained=True, layer=None):
        super(VisionTransformer_F, self).__init__()
        model_vit = vit_base_patch16_224_in21k(pretrained=pretrained)
        self.cls_token = model_vit.cls_token
        self.pos_embed = model_vit.pos_embed
        self.pos_drop = model_vit.pos_drop
        self.patch_embed = model_vit.patch_embed
        self.blocks = model_vit.blocks
        self.norm = model_vit.norm
        self.i = layer
        if layer is not None:
            self.__in_features = model_vit.embed_dim * 2
        else:
            self.__in_features = model_vit.embed_dim

    def _pos_embed(self, x):
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        if self.i is not None:
            f1 = self.blocks[:self.i](x)
            f2 = self.blocks[self.i:](f1)
            f2 = self.norm(f2)
            return f1, f2
        else:
            x = self.blocks(x)
            x = self.norm(x)
            return None, x

    @property
    def output_dim(self):
        return self.__in_features


class VisionTransformer(nn.Module):
    def __init__(self, pretrained=False, num_classes=1000, layer=None):
        super(VisionTransformer, self).__init__()
        self.backbone = VisionTransformer_F(pretrained=pretrained, layer=layer)
        self.norm = nn.LayerNorm(self.backbone.output_dim, eps=1e-6)
        self.head = nn.Linear(self.backbone.output_dim, num_classes)
        self.head.weight.data.normal_(0, 0.01)
        self.head.bias.data.fill_(0.0)

    def forward_feature(self, x):
        f1, f2 = self.backbone(x)
        if f1 is not None:
            f1 = f1[:, 1:].max(dim=1)[0]
            f2 = f2[:, 0]
            x = torch.concat((f1, f2), dim=-1)
        else:
            x = f2[:, 0]
        x = self.norm(x)
        return x

    def forward(self, x):
        feature = self.forward_feature(x)
        out = self.head(feature)
        return out


if __name__ == "__main__":
    net = VisionTransformer(pretrained=True, num_classes=5, layer=5)
    print(net)
