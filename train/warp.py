import torch
from einops import rearrange

def warp_torch(x, flo, mode='bilinear'):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, (T), C, H, W] (im2)
    flo: [B, (T) ,2, H, W] flow
    """
    if x.dim() == 5:
        b= x.shape[0]
        x =  rearrange(x, 'b f c h w -> (b f) c h w')
        reshape=True
    else: reshape=False

    if flo.dim() == 5:
        flo =  rearrange(flo, 'b f c h w -> (b f) c h w')

    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    mask = torch.ones(x.size(), dtype=x.dtype)
    if x.is_cuda:
        grid = grid.cuda()
        mask = mask.cuda()

    # flo = torch.flip(flo, dims=[1])
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(x, vgrid, align_corners=True, mode=mode)

    mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=True, mode=mode)


    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    out = output * mask

    if reshape:
        out = rearrange(out, '(b f) c h w -> b f c h w', b=b)

    return out