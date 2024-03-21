# GMARAFT: Non-Rigid Motion Estimation in Fully Sampled and Accelerated MR Images

This repository contains the Pytorch implementation of the paper
```
Aya Ghoul, Jiazhen Pan, Andreas Lingg, Jens Kübler, Patrick Krumm, Kerstin Hammernik, Daniel Rueckert, Sergios Gatidis, Thomas Küstner
"Attention-aware non-rigid image registration for accelerated MR imaging" (under review)
```
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8-ff69b4.svg" /></a>
<a href= "https://pytorch.org/"> <img src="https://img.shields.io/badge/PyTorch-1.13-2BAF2B.svg" /></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

GMARAFT is a deep learning-based method for the alignment/registration of both fully sampled and accelerated magnetic resonance (MR) images. We combined [RAFT](https://github.com/princeton-vl/RAFT), [GMA](https://github.com/zacjiang/GMA) and a denoising network to derive reliable motion estimation across different sampling trajectories (Cartesian and radial) and acceleration factors of up to 16x for cardiac motion and 30x for respiratory motion. The model was tested on the downstream task of motion-compensated reconstruction.

## Architecture

![GMARAFT](https://github.com/lab-midas/GMARAFT/blob/master/results/architecture.png)
Figure 1: Illustration of GMARAFT for non-rigid image registration with exemplary Cartesian VISTA undersampling.

## Usage

For a detailed description of the required conda environment, please refer to the provided requirements file. Additionally, the implementation relies on the Merlin (Machine Enhanced Reconstruction Learning and Interpretation Networks) library. Refer to [this link](https://github.com/midas-tum/merlin) for build instructions. Trained networks from the above manuscript can be obtained from the corresponding author upon reasonable request.

## Undersampling

The model's robustness was tested against three undersampling masks
- [Cartesian (VISTA)](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/mrm.25507)
- [2D golden angle radial undersampling](https://ieeexplore.ieee.org/abstract/document/4039540)
- [3D variable-density Poisson-Disc](https://ieeexplore.ieee.org/document/7486011)

## Training

To train the model, creating the data list in the loader needs to be first customized. Then, the config .json file can be edited to adjust training hyperparameters and paths. The run-pairwise.py script will train an image-to-image registration network (described in the paper). Model weights will be saved to a path specified in the config .json file. 

## Evaluation

To assess the model's quality by computing the photometric error between a fixed and warped test image, simply customize the eval.py script with the appropriate path to the model's weights and image data.

## Baseline Models for comparison
- [Elastix](https://github.com/SuperElastix/elastix)
- [VoxelMorph](https://github.com/voxelmorph/voxelmorph)
- [Vit-v-net](https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch)
- [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/tree/main)
- [XMorpher](https://github.com/Solemoon/XMorpher)

## Results

### Cardiac Motion Estimation for the fully sampled case
![R1_quiver](https://github.com/lab-midas/GMARAFT/blob/master/results/R1_quiver.png)
Figure 2: Motion estimation for relaxation (top) and contraction (bottom) motion in a representative test subject using Elastix, VoxelMoprh, Vit-v-net, TransMorph, XMorpher and our proposed method in a fully sampled case. VoxelMorph, Vit-V-Net, TransMorph and XMorpher are trained using fully sampled images. Color-encoded motion estimations (top row) are shown together with the moving images overlayed with quiver plots (bottom row). $t$ indicates the index of the image's time frame.
### Motion-compensated reconstructions for the fully sampled case
![R1_error](https://github.com/lab-midas/GMARAFT/blob/master/results/R1_results.png)
Figure 3: Motion-compensated reconstructions of the end-systolic and end-diastolic phase are shown in a fully sampled case for a representative subject. Images were reconstructed using iterative Sense with Elastix, VoxelMorph, Vit-V-Net, TransMorph, XMorpher and the proposed model motion estimates in comparison to the fully sampled ground truth (GT) image. VoxelMorph, Vit-V-Net, TransMorph and XMorpher are trained using fully sampled images.
### Cardiac Motion Estimation for accelerated data
![CINE_flow_estimation](https://github.com/lab-midas/GMARAFT/blob/master/results/CINE_flow_estimation.png)
Figure 4: Cardiac motion estimation is shown for the fully sampled and retrospectively undersampled acquisitions with VISTA (Cartesian) and radial (non-Cartesian) undersampling for R=16 acceleration.
### Respiratory Motion Estimation for accelerated data
<img src="https://github.com/lab-midas/GMARAFT/blob/master/results/Resp_flow_estimation.png" width="600" />

Figure 5: Respiratory motion estimation for the fully sampled and Cartesian vdPD accelerated acquisitions with accelerations R=16 and R=30. Deformation fields are overlaid on the moving image. Images of motion-compensated reconstructions are depicted next to the used color-encoded deformation fields.
### Quantitative results compared to baselines
1. Registration performance
![photometric_loss](https://github.com/lab-midas/GMARAFT/blob/master/results/photometric_loss.png)
Figure 6: Violin plots of the residual photometric error values in fully-sampled, R=8 and R=16 test data for VISTA and radial undersampling.

![Jac_det](https://github.com/lab-midas/GMARAFT/blob/master/results/Jac_det.png)
Figure 7: Violin plots of the percentages of non-positive values in the determinant of the Jacobian matrix values in fully-sampled, R=8 and R=16 test data for VISTA and radial undersampling.


3. Downstream task: Motion-compensated reconstruction image quality metrics
![SSIM](https://github.com/lab-midas/GMARAFT/blob/master/results/SSIM.png)
Figure 8: Violin plots of the SSIM in fully-sampled, R=8 and R=16 test data for VISTA and radial undersampling.

![PSNR](https://github.com/lab-midas/GMARAFT/blob/master/results/PSNR.png)
Figure 9: Violin plots of the PSNR in fully-sampled, R=8 and R=16 test data for VISTA and radial undersampling.

![NRMSE](https://github.com/lab-midas/GMARAFT/blob/master/results/NRMSE.png)
Figure 10: Violin plots of the NRMSE in fully-sampled, R=8 and R=16 test data for VISTA and radial undersampling.

![HFEN](https://github.com/lab-midas/GMARAFT/blob/master/results/HFEN.png)
Figure 11: Violin plots of the HFEN in fully-sampled, R=8 and R=16 test data for VISTA and radial undersampling.

