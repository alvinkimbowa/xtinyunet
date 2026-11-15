"""PlainConvUNet whose decoder width is set separately from the encoder width.

nnU-Net's PlainConvUNet sizes every decoder stage from the matching encoder stage. Here the
decoder stages take their widths from `features_per_stage_decoder` instead, so a plans config can
narrow the encoder while keeping the decoder. Plans select it with
`"network_class_name": "asym_unet.HalfEncoderUNet"`, which needs `src` on PYTHONPATH.
"""

import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks


class AsymUNetDecoder(nn.Module):
    """UNetDecoder with its own per-stage widths, ordered like the encoder (full resolution first).

    `decoder_features[-1]` is unused: the bottleneck is the last encoder stage.
    """

    def __init__(self, encoder, decoder_features, num_classes, n_conv_per_stage, deep_supervision, nonlin_first=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        enc = encoder.output_channels
        n = len(enc)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n - 1)
        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        stages, transpconvs, seg_layers = [], [], []
        below = enc[-1]
        for s in range(1, n):
            skip = enc[-(s + 1)]
            out = decoder_features[-(s + 1)]
            stride = encoder.strides[-s]
            transpconvs.append(transpconv_op(below, out, stride, stride, bias=encoder.conv_bias))
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s - 1], encoder.conv_op, out + skip, out,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.conv_bias,
                encoder.norm_op, encoder.norm_op_kwargs, encoder.dropout_op, encoder.dropout_op_kwargs,
                encoder.nonlin, encoder.nonlin_kwargs, nonlin_first,
            ))
            seg_layers.append(encoder.conv_op(out, num_classes, 1, 1, 0, bias=True))
            below = out

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        x = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](x)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == len(self.stages) - 1:
                seg_outputs.append(self.seg_layers[-1](x))
        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]


class HalfEncoderUNet(PlainConvUNet):
    def __init__(self, *args, features_per_stage_decoder, n_conv_per_stage_decoder, deep_supervision=False,
                 nonlin_first=False, **kwargs):
        super().__init__(*args, n_conv_per_stage_decoder=n_conv_per_stage_decoder,
                         deep_supervision=deep_supervision, nonlin_first=nonlin_first, **kwargs)
        self.decoder = AsymUNetDecoder(self.encoder, features_per_stage_decoder, self.decoder.num_classes,
                                       n_conv_per_stage_decoder, deep_supervision, nonlin_first)
