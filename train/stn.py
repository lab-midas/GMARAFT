import torch
import torch.nn.functional as F
from einops import rearrange

class SpatialTransformer(torch.nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, src, flow, mode='bilinear'):
        if src.dim() == 5:
            [b_1, f_1, c_1, h_1, w_1] = src.shape
            src = torch.reshape(src, (b_1 * f_1, c_1, h_1, w_1))
        if flow.dim() == 5:
            [b, f, c, h, w] = flow.shape
            flow = torch.reshape(flow, (b * f, c, h, w))
            reshape = True
        else:
            reshape = False

        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        grid = grid.cuda()
        # grid = grid

        new_locs = grid + flow

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        out = F.grid_sample(src, new_locs, mode=mode)

        if reshape:
            out = rearrange(out, '(b f) c h w -> b f c h w', b=b)

        return out