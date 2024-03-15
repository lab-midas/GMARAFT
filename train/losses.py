import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math


class EAE(nn.Module):
    def __init__(self):
        super(EAE, self).__init__()

    def forward(self, inputs, targets):
        EAE_loss = torch.acos((1 + torch.sum(targets * inputs)) /
                              (torch.sqrt(1 + torch.sum(torch.pow(inputs, 2))) *
                               torch.sqrt(1 + torch.sum(torch.pow(targets, 2)))))

        return EAE_loss


class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, inputs, targets):
        EPE_loss = torch.mean(torch.square(targets - inputs))
        return EPE_loss


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def smooth_grad_1st(flow, image=None, boundary_awareness=False, alpha=None):
    dx, dy = gradient(flow)
    dx, dy = dx.abs(), dy.abs()
    if boundary_awareness:
        img_dx, img_dy = gradient(image)
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)
        loss_x = weights_x * dx / 2.
        loss_y = weights_y * dy / 2.
    else:
        loss_x = dx
        loss_y = dy
    return loss_x.mean() / 2. + loss_y.mean() / 2.


def smooth_grad_2nd(flo, image=None, boundary_awareness=False, alpha=None):
    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    eps = 1e-6
    dx2, dy2 = torch.sqrt(dx2 ** 2 + eps), torch.sqrt(dy2 ** 2 + eps)
    if boundary_awareness:
        img_dx, img_dy = gradient(image)
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)
        loss_x = weights_x[:, :, :, 1:] * dx2
        loss_y = weights_y[:, :, 1:, :] * dy2
    else:
        loss_x = dx2
        loss_y = dy2
    return loss_x.mean() / 2. + loss_y.mean() / 2.


class Grad2D(nn.Module):
    def __init__(self, mode=1, boundary_awareness=False, alpha=10):
        super(Grad2D, self).__init__()
        self.boundary_awareness = boundary_awareness
        self.alpha = alpha
        assert mode in [1, 2]
        if mode == 2:
            self.func_smooth = smooth_grad_2nd
        elif mode == 1:
            self.func_smooth = smooth_grad_1st

    def forward(self, flow_vec, image):
        return self.func_smooth(flow_vec, image, self.boundary_awareness, self.alpha).mean()


class PhotometricLoss(torch.nn.Module):
    def __init__(self, mode='L1'):
        super(PhotometricLoss, self).__init__()
        assert mode in ('L1', 'L2')
        self.mode = mode
        if mode == 'L1':
            self.loss = torch.nn.L1Loss(reduction='mean')
        elif mode == 'L2':
            self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, inputs, outputs):
        if self.mode == 'L1':
            return (inputs - outputs).abs().mean()

        return self.loss(inputs, outputs)

