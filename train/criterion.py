import torch
from .losses import *
from .warp import warp_torch
from .stn import SpatialTransformer


class CriterionBase(torch.nn.Module):
    def __init__(self, config):
        super(CriterionBase, self).__init__()
        self.loss_names = config['loss_names']
        self.loss_weights = config['loss_weights']
        self.iteration_gamma = config['iteration_gamma']
        self.loss_args = config['loss_args']
        self.loss_list = []
        for loss_name in self.loss_names:
            loss_args = config['loss_args'][loss_name]
            loss_item = self.get_loss(loss_name=loss_name, args_dict=loss_args)
            self.loss_list.append(loss_item)

    def get_loss(self, loss_name, args_dict):
        if loss_name == 'photometric':
            return PhotometricLoss(**args_dict)
        elif loss_name == 'smooth':
            return Grad2D(**args_dict)
        elif loss_name == 'temporal_smooth':
            return TemporalSmooth(**args_dict)
        elif loss_name == 'denoiser':
            return torch.nn.MSELoss()
        else:
            raise NotImplementedError


class Criterion(CriterionBase):
    def __init__(self, config):
        super().__init__(config=config)
        self.iterative_losses: list = config['iterative_losses']
        self.iteration_gamma: float = config['iteration_gamma']
        self.diffeomorphic: bool = config['diffeomorphic']
        if self.diffeomorphic:
            self.diffeo_weight: float = config['diffeo_weight']
            self.diffeomorphic_mse = torch.nn.MSELoss()
            self.stn = SpatialTransformer()

    def forward(self, y_pred, y_true):
        loss_dict = {}
        total_loss = 0
        flow_preds, context_up = y_pred
        ref, mov, context = [x.cuda() for x in y_true]
        total_iters = len(flow_preds)
        for loss_name, loss_weight, loss_term in zip(self.loss_names, self.loss_weights, self.loss_list):
            partial_loss = 0
            if loss_name in self.iterative_losses:
                i_weights = [self.iteration_gamma ** (total_iters - iteration - 1) for iteration in range(total_iters)]
                flow_in_loop = flow_preds.copy()
            else:
                i_weights = [1]
                flow_in_loop = [flow_preds[-1]]
            for i, (i_weight, i_flow) in enumerate(zip(i_weights, flow_in_loop)):
                if loss_name == 'photometric':
                    warped = warp_torch(ref, i_flow)
                    i_loss = loss_term(mov, warped)
                elif loss_name == 'smooth':
                    i_loss = loss_term(i_flow, ref)
                elif loss_name == 'temporal_smooth':
                    i_loss = loss_term(i_flow)
                elif loss_name == 'denoiser':
                    i_loss = loss_term(context, context_up)
                else:
                    raise KeyError('loss_name not defined')
                partial_loss += i_weight * loss_weight * i_loss

            loss_dict[loss_name] = partial_loss
            total_loss += partial_loss

        if self.diffeomorphic:
            for loss_name, loss_weight, loss_term in zip(self.loss_names, self.loss_weights, self.loss_list):
                partial_loss = 0
                if loss_name in self.iterative_losses:
                    i_weights = [self.iteration_gamma ** (total_iters - iteration - 1) for iteration in range(total_iters)]
                    flow_predictions = flow_preds.copy()
                    flow_predictions_backward = [-1 * self.stn(flow, flow) for flow in flow_predictions]

                    for i, (i_weight, i_flow) in enumerate(zip(i_weights, flow_in_loop)):
                        if loss_name == 'photometric':
                            warped = warp_torch(ref, i_flow)
                            i_loss = loss_term(mov, warped)
                        elif loss_name == 'smooth':
                            i_loss = loss_term(i_flow, ref)
                        elif loss_name == 'temporal_smooth':
                            i_loss = loss_term(i_flow)
                        else:
                            raise KeyError('loss_name not defined')
                        partial_loss += i_weight * loss_weight * i_loss

                    loss_dict[f'{loss_name}_diffeo'] = partial_loss
                    total_loss += partial_loss

            loss_dict['diffeo'] = self.diffeomorphic_mse(flow_preds[-1] + flow_predictions_backward[-1], torch.zeros_like(flow_predictions_backward[-1]))
            total_loss += self.diffeo_weight * loss_dict['diffeo']

        loss_dict['total_loss'] = total_loss
        return loss_dict