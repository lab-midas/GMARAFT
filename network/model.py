import torch
import torch.nn as nn
import torch.nn.functional as F
from .update import GMAUpdateBlock
from .encoder import BasicEncoder
from .corr import CorrBlock
from .attention import Attention
from .denoiser import ResNet
from .utils import coords_grid, upflow4
autocast = torch.cuda.amp.autocast

class RAFTBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_channels = 1
        self.num_heads = 1
        self.iters = 6
        self.corr_radius = 4
        self.hidden_dim = 128
        self.context_dim = 128

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img, factor=4):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // factor, W // factor).to(img.device)
        coords1 = coords_grid(N, H // factor, W // factor).to(img.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/4, W/4, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 4 * H, 4 * W)

class GMARAFT_Denoiser(RAFTBase):
    def __init__(self):
        super(GMARAFT_Denoiser, self).__init__()
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', num_input_channels=self.num_channels)
        self.cnet = BasicEncoder(output_dim=self.hidden_dim + self.context_dim, norm_fn='batch', num_input_channels=3)
        self.update_block = GMAUpdateBlock(hidden_dim=self.hidden_dim)
        self.att = Attention(dim=self.context_dim, heads=self.num_heads, max_pos_size=160, dim_head=self.context_dim)
        self.resnet = ResNet(in_channels=3)

    def forward(self, image1, image2, context_image, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """
        reshape=False
        if image1.dim() == 5:
            reshape=True
            [b, f, _, h, w] = image1.shape
            image1 = torch.reshape(image1, (b * f, 1, h, w))
            image2 = torch.reshape(image2, (b * f, 1, h, w))
            context_image = torch.reshape(context_image, (b * f, 3, h, w))

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        context_image = 2 * (context_image / 255.0) - 1.0
        context_image = context_image.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=True):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        with autocast(enabled=True):
            context_image_up = self.resnet(context_image)
            context_image_up = 2 * ((context_image_up - context_image_up.min()) / (context_image_up.max() - context_image_up.min())) - 1
            cnet = self.cnet(context_image_up)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)
        # print(coords0.shape)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(self.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
            # print(corr.shape)

            flow = coords1 - coords0
            with autocast(enabled=True):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            if up_mask is None:
                flow_up = upflow4(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)

        if test_mode:
            if reshape:
                flow_up = torch.reshape(flow_up, (b, f, 2, h, w))
            return coords1 - coords0, flow_up, context_image_up

        if reshape:
            context_image_up = torch.reshape(context_image, (b, f, 3, h, w))

        return flow_predictions, context_image_up