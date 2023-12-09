from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
from .attention import SelfAttnPropagation
from attention import PositionEncodingSine, LocalFeatureTransformer


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 2, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 3, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)    #1/4
        l2 = self.layer2(x)   #1/8
        l3 = self.layer3(l2)  #1/16
        l4 = self.layer4(l3)  #1/16

        # Interpolate all feature maps to the same size, namely the size of the smallest feature map
        h_min, w_min = min(l2.shape[2:]), min(l2.shape[3:])

        #l2 = F.interpolate(l2, size=(h_min, w_min), mode='bilinear', align_corners=True)
        l3 = F.interpolate(l3, size=(h_min, w_min), mode='bilinear', align_corners=True)
        l4 = F.interpolate(l4, size=(h_min, w_min), mode='bilinear', align_corners=True)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)
        return gwc_feature


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d_dw(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(EfficientDWConv3d(in_channels * 2, 3, 3, 0.125),
                                   nn.BatchNorm3d(in_channels * 2),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d_dw(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(EfficientDWConv3d(in_channels * 4, 3, 3, 0.125),
                                   nn.BatchNorm3d(in_channels * 4),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)#64*1/8

        conv3 = self.conv3(conv2)#128*1/16
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class EfficientDWConv3d(nn.Module):


    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=7, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_htw = nn.Conv3d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_wd = nn.Conv3d(gc, gc, kernel_size=(1, band_kernel_size, band_kernel_size),
                                   padding=(0, band_kernel_size // 2, band_kernel_size // 2), groups=gc)
        self.dwconv_hd = nn.Conv3d(gc, gc, kernel_size=(band_kernel_size, 1, band_kernel_size),
                                   padding=(band_kernel_size // 2, 0, band_kernel_size // 2), groups=gc)
        self.dwconv_hw = nn.Conv3d(gc, gc, kernel_size=(band_kernel_size, band_kernel_size, 1),
                                   padding=(band_kernel_size // 2, band_kernel_size // 2, 0), groups=gc)
        self.split_indexes = (in_channels - 4 * gc, gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_htw, x_wd, x_hd, x_hw = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_htw(x_htw), self.dwconv_wd(x_wd), self.dwconv_hd(x_hd), self.dwconv_hw(x_hw)),
            dim=1,
        )

class GPDFNet(nn.Module):
    def __init__(self, maxdisp):
        super(GPDFNet, self).__init__()
        self.maxdisp = maxdisp
        self.concat_channels = 32
        self.feature_extraction = feature_extraction()
        self.concatconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, self.concat_channels, kernel_size=1, padding=0, stride=1,
                                                  bias=False))

        self.dres1_att_ = nn.Sequential(convbn_3d(40, 32, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        convbn_3d(32, 32, 3, 1, 1))
        self.dres2_att_ = hourglass(32)
        self.classif_att_ = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # propagation with self-attn
        self.feature_flow_attn = SelfAttnPropagation(in_channels=128)


        self.upsampler = nn.Sequential(nn.Conv2d(130, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 4 ** 2 * 9, 1, 1, 0))

        self.self_att_fn = LocalFeatureTransformer(
            d_model=320, nhead=8, layer_names=["self"] * 1, attention="linear"
        )
        self.cross_att_fn = LocalFeatureTransformer(
            d_model=320, nhead=8, layer_names=["cross"] * 1, attention="linear"
        )

        self.dres = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                  nn.ReLU(inplace=True),
                                  convbn(128, 128, 3, 1, 1, 1),
                                  nn.ReLU(inplace=True)
                                  )
        self.dres0 = nn.Sequential(convbn_3d_dw(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   EfficientDWConv3d(32, 3, 3, 0.125),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(EfficientDWConv3d(32, 3, 3, 0.125),
                                   nn.ReLU(inplace=True),
                                   EfficientDWConv3d(32, 3, 3, 0.125))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(EfficientDWConv3d(32, 3, 3, 0.125),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(EfficientDWConv3d(32, 3, 3, 0.125),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(EfficientDWConv3d(32, 3, 3, 0.125),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(EfficientDWConv3d(32, 3, 3, 0.125),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=4,
                      is_depth=False):
        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * multiplier
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            up_flow = upsample_flow_with_mask(flow, mask, upsample_factor=self.upsample_factor,
                                              is_depth=is_depth)

        return up_flow

    def forward(self, left, right):

        features_left_c = self.feature_extraction(left)
        features_right_c = self.feature_extraction(right)

        #Geometry prior guidance module
        a = features_left_c.size(2)
        # num_att = 2
        # positional encoding and self-attention and cross-attention
        pos_encoding_fn_small = PositionEncodingSine(
            d_model=320, max_shape=(features_left_c.shape[2], features_left_c.shape[3])
        )
        # 'n c h w -> n (h w) c'
        x_tmp = pos_encoding_fn_small(features_left_c)
        features_left = x_tmp.permute(0, 2, 3, 1).reshape(x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3],
                                                          x_tmp.shape[1])
        # 'n c h w -> n (h w) c'
        x_tmp = pos_encoding_fn_small(features_right_c)
        features_right = x_tmp.permute(0, 2, 3, 1).reshape(x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3],
                                                           x_tmp.shape[1])

        features_left, features_right = self.self_att_fn(features_left, features_right)
        features_left, features_right = self.cross_att_fn(features_left, features_right)

        features_left_t, features_right_t = [
            x.reshape(x.shape[0], a, -1, x.shape[2]).permute(0, 3, 1, 2)
            for x in [features_left, features_right]
        ]

        features_left_t = F.interpolate(features_left_t, scale_factor=2,
                                        mode='bilinear', align_corners=True) * 2
        features_right_t = F.interpolate(features_right_t, scale_factor=2,
                                         mode='bilinear', align_corners=True) * 2
        # B, C, H, W = features_left_t.shape
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # dres00 = nn.Sequential(convbn_3d(W, 48, 3, 1, 1),
        #                        nn.ReLU(inplace=True),
        #                        convbn_3d(48, 48, 3, 1, 1),
        #                        nn.ReLU(inplace=True)).to(device)

        features0 = features_left_t
        features0 = self.dres(features0)

        #Concatenation volume construction
        concat_feature_left = self.concatconv(features_left_t)
        concat_feature_right = self.concatconv(features_right_t)
        concat_volume = build_concat_volume(concat_feature_left, concat_feature_right,
                                            self.maxdisp // 4)  # [B,C,128,H,W]

        #Disparity feature generation and cost volume refinement
        disp_pred, prob = global_correlation_softmax_stereo(features_left_t,
                                                            features_right_t)  # prob.shape[B,H,W/4,W/4], disp_pred.shape[B,1,H/4,W/4]
        disp_pred = disp_pred.clamp(min=0)
        disp_pred = self.feature_flow_attn(features0, disp_pred.detach(),
                                           local_window_attn=False,
                                           local_window_radius=1,
                                           )

        DF = disp_pred.unsqueeze(1)  #disparity feature
        DFVolume = torch.cat((concat_volume,DF))

        cost0 = self.dres0(DFVolume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)
        


        if self.training:

            pred_attention = self.upsample_flow(disp_pred, features0, bilinear=True,
                                                upsample_factor=4,
                                                is_depth=False)
            pred_attention = torch.squeeze(pred_attention, 1)

            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)

            return [pred_attention, pred0, pred1, pred2,pred3]

        else:

            
            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)

            return [pred3]


def gpdf(d):
    return GPDFNet(d)



