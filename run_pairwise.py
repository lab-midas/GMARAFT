from network.model import GMARAFT_Denoiser
from train.trainer import Trainer
from loader.loader_cine import CineDatasetPairwise
import json
import torch.utils.data as data
import os

json_file_path = "/path_to/configs/train_cine_group.json"
with open(json_file_path, 'r') as file:
    config = json.load(file)


## load model
model = GMARAFT_Denoiser()
model.cuda()
model.train()

## read data
mode = 'debug' if config['debug'] else 'train'
train_dataset = CineDatasetPairwise(config['data_loader'], mode=mode)
train_loader = data.DataLoader(train_dataset,
                                   batch_size=config['data_loader']['batch_size'],
                                   pin_memory=True,
                                   shuffle=True,
                                   num_workers=config['data_loader']['num_workers'],
                                   drop_last=True)
print('Loader has %d cine image pairs' % len(train_dataset))
print('steps per epoch', len(train_loader))

## run training
trainer = Trainer(config, model=model, data_loader=train_loader)
trainer.run()
