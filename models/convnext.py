import torch
import timm

from torch import nn
from torch import Tensor
from typing import List



class ConvNormAct(nn.Sequential):
    """
    A little util layer composed by (conv) -> (norm) -> (act) layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm=nn.BatchNorm2d,
        act=nn.ReLU,
        **kwargs
    ):
        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs,
            ),
            norm(out_features),
            act(),
        )


class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(
            init_value * torch.ones((dimensions)), requires_grad=True
        )

    def forward(self, x):
        return self.gamma[None, ..., None, None] * x


class BottleNeckBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 4,
        drop_p: float = 0.0,
        layer_scaler_init_value: float = 1e-6,
    ):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = nn.Sequential(
            # narrow -> wide (with depth-wise and bigger kernel)
            nn.Conv2d(
                in_features,
                in_features,
                kernel_size=7,
                padding=3,
                bias=False,
                groups=in_features,
            ),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            # wide -> wide
            nn.Conv2d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            # wide -> narrow
            nn.Conv2d(expanded_features, out_features, kernel_size=1),
        )
        self.layer_scaler = LayerScaler(layer_scaler_init_value, out_features)
        

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x = self.layer_scaler(x)
        # x = self.drop_path(x)
        x += res
        return x


class ConvNexStage(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, depth: int, **kwargs):
        super().__init__(
            # add the downsampler
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=in_features),
                nn.Conv2d(in_features, out_features, kernel_size=2, stride=2),
            ),
            *[
                BottleNeckBlock(out_features, out_features, **kwargs)
                for _ in range(depth)
            ],
        )


class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=4),
            nn.BatchNorm2d(out_features),
        )


class ConvNextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        depths: List[int],
        widths: List[int],
        drop_p: float = 0.0,
    ):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features)

        in_out_widths = list(zip(widths, widths[1:]))
        # create drop paths probabilities (one for each stage)
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))]

        self.stages = nn.ModuleList(
            [
                ConvNexStage(stem_features, widths[0], depths[0], drop_p=drop_probs[0]),
                *[
                    ConvNexStage(in_features, out_features, depth, drop_p=drop_p)
                    for (in_features, out_features), depth, drop_p in zip(
                        in_out_widths, depths[1:], drop_probs[1:]
                    )
                ],
            ]
        )

    def forward(self, x):
        x = x.float()
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


class ClassificationHead(nn.Sequential):
    def __init__(self, num_channels: int, num_classes: int = 1000):
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_classes),
            
        )


class ConvNext(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        depths: List[int],
        widths: List[int],
        drop_p: float = 0.0,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.encoder = ConvNextEncoder(
            in_channels, stem_features, depths, widths, drop_p
        )
        self.head = ClassificationHead(widths[-1], num_classes)


class ConvNext_backbone_model(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.model = timm.create_model('convnext_base', pretrained=True, in_chans=in_channels)
        self.fc = nn.Sequential(
                                nn.Linear(1024 * 3 * 3, 1024),
                                nn.PReLU(),
                                nn.BatchNorm1d(1024),
                                nn.Dropout(p = 0.5),
                                
                                nn.Linear(1024, 512),
                                nn.BatchNorm1d(512),
                                nn.PReLU(),
                                nn.Dropout(p = 0.5),
        
                                nn.Linear(512, 128),
                                nn.PReLU(),
                                nn.BatchNorm1d(128),
                                nn.Dropout(p = 0.3),
                                
                                nn.Linear(128, num_classes)
                                )

    def forward(self,x):
        x = self.model.forward_features(x)
        
        B, C, H, W = x.shape
        x = x.reshape((B, C*H*W))
        x = self.fc(x)
        return x

if __name__ == "__main__":
    image = torch.rand(2, 1, 100, 100)
    classifier = ConvNext_backbone_model(in_channels=1,num_classes=3)

    print(classifier(image).shape)
