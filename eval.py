import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import h5py
from network.model import GMARAFT_Denoiser
from train.warp import warp_torch
from train.losses import PhotometricLoss
from evaluate.utils import add_quiver, increase_brightness, get_data
import flow_vis

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

## load checkpoint
model = GMARAFT_Denoiser()
model.cuda()
model.eval()
checkpoint_path = '/path/to/checkpoint'
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)

### load data
R = 16
t1, t2 = 5, 12
slice = 5
mode = "pairwise" # or "groupwise
filename = 'Sub0094'

with h5py.File(f"/path/toh5/h5/{filename}.h5", 'r') as f:
    img_fully = np.abs(f['dImgC'][:])

img_R = np.load(f'/pathto/VISTA/R{R}/{filename}_img.npy')
img_ref, img_mov, context_img = get_data(img_R, slice, t1, t2, mode)
img_ref_fully, img_mov_fully, context_img_fully = get_data(img_fully, slice, t1, t2, mode)

## predict
with torch.no_grad():
    flow_low, flow_pr, context_image_up = model(torch.from_numpy(img_ref[None,None]).float().cuda(),
                                                torch.from_numpy(img_mov[None,None]).float().cuda(),
                                                torch.from_numpy(context_img[None]).float().cuda(),
                                                test_mode=1)
    warped_th = warp_torch(img_mov_fully[None,None].cuda(), flow_pr)
    warped = warped_th.cpu().detach().numpy()[0,0]
flow = np.transpose(flow_pr[0].cpu().numpy(), (1, 2, 0))
flow_img = flow_vis.flow_to_color(flow, convert_to_bgr=False)

## plot
fig, axes = plt.subplots(1, 7, figsize=(10, 2))
font = 10
axes[0].set_title(f'I_ref x{R}', fontsize=font)
axes[1].set_title(f'I_mov x{R}', fontsize=font)
axes[2].set_title('Moving+flow\n(fully-sampled)', fontsize=font)
axes[3].set_title('Moving warped\n(fully-sampled)', fontsize=font)
axes[4].set_title('Moving-Ref', fontsize=font)
axes[5].set_title('Warped-Ref', fontsize=font)
axes[6].set_title('Prediction', fontsize=font)

axes[0].imshow(img_ref, cmap='gray')
axes[1].imshow(img_mov, cmap='gray')
axes[2].imshow(increase_brightness(img_mov_fully), cmap='gray')
add_quiver(axes[0][2], flow, 8, scale=40)
axes[3].imshow(increase_brightness(warped), cmap='gray')
axes[4].imshow(img_ref-img_mov, cmap='RdBu', vmin=-0.5, vmax=0.5)
axes[5].imshow(img_ref-warped, cmap='RdBu', vmin=-0.5, vmax=0.5)
axes[6].imshow(flow_img, fontsize=font)

for i in range(7):
    axes[i].axis('off')
plt.show()


### metrics
photometric_loss = PhotometricLoss()
photo_before_warping =  photometric_loss(torch.from_numpy(img_ref_fully), torch.from_numpy(img_mov_fully) )
photometric_after_warping = photometric_loss(torch.from_numpy(img_ref_fully), torch.from_numpy(warped_th) )
print(f'#########\n photo_before_warping: {photo_before_warping} \n photometric_after_warping: {photometric_after_warping}')

