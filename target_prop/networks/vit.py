from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from simple_parsing.helpers import list_field

from target_prop.backward_layers import invert
from target_prop.layers import AdaptiveAvgPool2d, Reshape
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from .network import Network


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 64):
        self.patch_size = patch_size
        super().__init__()
        self.emb_size = emb_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.img_size = img_size
        # using a conv layer instead of a linear one -> performance gains
        self.conv = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.rearrange = Rearrange('b e (h) (w) -> b (h w) e')
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        print(x.shape)
        b, _, _, _ = x.shape
        x = self.conv(x)
        print(x.shape)
        x = self.rearrange(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x

class InvertPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 64):
        self.patch_size = patch_size
        super().__init__()

        # using a conv layer instead of a linear one -> performance gains
        self.conv = invert(nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size))
        self.rearrange = Rearrange(' b (h w) e ->  b e (h) (w)',h=img_size//patch_size)

    def forward(self, x):
        x  = x[:,1:,:]
        x = self.rearrange(x)
        print(self.conv)
        print(x.shape)
        x = self.conv(x)
        print(x.shape)
        return x

#Backward net will not train. This is just to ensure that size is consistent
@invert.register(PatchEmbedding)
def invert_basic(module: PatchEmbedding) -> InvertPatchEmbedding:
    backward = InvertPatchEmbedding(
        emb_size = module.emb_size,
        in_channels = module.in_channels,
        patch_size = module.patch_size,
        img_size = module.img_size,
    )
    return backward


class EncoderBasicBlock(nn.Module):
    """"
    Simple ViT Encoder Block
    """
    def __init__(self,emb_size: int = 768, drop_p: float = 0., forward_expansion: int = 4,forward_drop_p: float = 0.,num_heads=8):
        super(EncoderBasicBlock,self).__init__()
        self.emb_size = emb_size
        self.drop_p = drop_p
        self.num_heads = num_heads
        self.forward_expansion = forward_expansion
        self.forward_drop_p = forward_drop_p
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(drop_p)

        self.linear0 = nn.Linear(emb_size,emb_size*3)
        self.attn = nn.MultiheadAttention(emb_size,num_heads,batch_first=True)
        self.linear1 = nn.Linear(emb_size,forward_expansion*emb_size)
        self.linear2 = nn.Linear(emb_size*forward_expansion, emb_size)

    def forward(self,x):
        attout = self.ln1(x)
        # attout = self.layer0(attout) #split qkv, maybe not necessary
        attout,_ = self.attn(attout,attout,attout)

        # x = self.dropout(attout) + x
        x = attout+x
        fout   = self.ln2(x)
        fout   = self.linear1(fout)
        fout   = F.gelu(fout)
        # fout   = self.dropout(fout)
        fout   = self.linear2(fout)
        out = fout + x
        return out

class InvertEncoderBasicBlock(nn.Module):
    """"
    ViT Encoder Invert. Need to test this.
    """
    def __init__(self,emb_size: int = 768, drop_p: float = 0., forward_expansion: int = 4,forward_drop_p: float = 0.,num_heads=8):
        super(InvertEncoderBasicBlock,self).__init__()
        self.emb_size = emb_size
        self.drop_p = drop_p
        self.forward_expansion = forward_expansion
        self.forward_drop_p = forward_drop_p
        self.ln1 = invert(nn.LayerNorm(emb_size))
        self.ln2 = invert(nn.LayerNorm(emb_size))
        self.dropout = invert(nn.Dropout(drop_p))
        self.linear0 = invert(nn.Linear(emb_size,emb_size*3))
        self.attn = invert(nn.MultiheadAttention(emb_size,num_heads,batch_first=True))
        self.linear1 = invert(nn.Linear(emb_size,forward_expansion*emb_size))
        self.linear2 = invert(nn.Linear(emb_size*forward_expansion, emb_size))

    def forward(self,x):
        fout = self.linear2(x)
        # fout = self.dropout(fout)
        fout = F.gelu(fout)
        fout = self.linear1(fout)
        fout = self.ln2(fout)
        # x += self.dropout(fout)
        attout,_  = self.attn(x,x,x)
        out = attout + x
        return self.ln1(out)

# Need to experiment with the inversion. Can we treat the entire Encoder as its own inversion?
@invert.register(EncoderBasicBlock)
def invert_basic(module: EncoderBasicBlock) -> InvertEncoderBasicBlock:
    backward = InvertEncoderBasicBlock(
        emb_size = module.emb_size,
        drop_p =module.drop_p,
        forward_expansion= module.forward_expansion,
        forward_drop_p = module.forward_drop_p,
        num_heads = module.num_heads
    )
    return backward



class ClassificationHead(nn.Module):
    def __init__(self, num_patches=16,emb_size: int = 768, n_classes: int = 1000):
        super(ClassificationHead,self).__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.num_patches = num_patches
        self.ln  = nn.LayerNorm(emb_size)
        self.lin = nn.Linear(emb_size,n_classes)
    def forward(self,x):
        out = x.mean(1)
        out = self.ln(out)
        out = self.lin(out)
        return out

class InverseClassificationHead(nn.Module):
    def __init__(self, num_patches=16,emb_size: int = 768, n_classes: int = 1000):
        super(InverseClassificationHead,self).__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.ln  = nn.LayerNorm(emb_size)
        self.lin = nn.Linear(emb_size,n_classes)
        self.num_patches = num_patches
        self.up  = nn.Upsample(scale_factor=num_patches+1) #num_patches +1 if using standard implementation

    def forward(self,x):
        out = self.lin(x)
        out = self.ln(out)
        out = self.up(out.unsqueeze(-1)).permute(0,2,1)
        return out

@invert.register(ClassificationHead)
def invert_basic(module: ClassificationHead) -> InverseClassificationHead:
    backward = InverseClassificationHead(
        num_patches = module.num_patches,
        emb_size    = module.emb_size,
        n_classes   = module.n_classes
    )
    return backward


class ViT(nn.Sequential, Network):
    @dataclass
    class HParams(Network.HParams):
        channels: List[int] = list_field(128, 128, 256, 256, 512)
        bias: bool = True

    def __init__(self, in_channels: int = 3,
                 patch_size: int = 8,
                 emb_size: int = 512,
                 img_size: int = 32,
                 depth: int = 2,
                 n_classes: int = 1000,
                 hparams:"ViT.HParams"= None):

        layers: OrderedDict[str, nn.Module] = OrderedDict()

        layers["layer_0"] = nn.Sequential(
            OrderedDict(
                patchemb=PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            )
        )

        for i in range(1,depth+1):
            layers["layer_"+str(i)] = EncoderBasicBlock(emb_size=512, drop_p=0, forward_expansion=4,forward_drop_p=0,num_heads=8)

        layers["fc"] = nn.Sequential(
            OrderedDict(
                pool=AdaptiveAvgPool2d(
                    output_size=(1, 1)
                ),  # NOTE: This is specific for 32x32 input!
                reshape=Reshape(target_shape=(-1,)),
                linear=nn.LazyLinear(out_features=n_classes, bias=True),
            )
        )
        super().__init__(layers)



vit = ViT
ViTHparams = ViT.HParams