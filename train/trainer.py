import torch
import pathlib
from .criterion import Criterion

class BaseTrainer:
    def __init__(self, config, model, data_loader):
        self.config = config
        self.data_loader = data_loader
        self.debug = config['debug']

        cfg_trainer = config['trainer']
        self.start_epoch = 0
        self.num_epochs = cfg_trainer['epochs']
        self.group = cfg_trainer['group']
        self.clip = cfg_trainer['clip']
        self.gamma = cfg_trainer['gamma']

        if not self.debug:
            wandb_setup(config)
            self.checkpoint_dir = cfg_trainer['save_dir'] + '/' + config['name']
            pathlib.Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


        self.model = model
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = torch.optim.AdamW(trainable_params,
                                           lr = config['optimizer']['args']['lr'],
                                           weight_decay = config['optimizer']['args']['weight_decay'],
                                           amsgrad=config['optimizer']['args']['amsgrad'],
                                           eps=config['optimizer']['args']['eps'])



        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                                                max_lr=0.00025,
                                                                steps_per_epoch=len(self.data_loader),
                                                                epochs=self.num_epochs,
                                                                pct_start=0.05,
                                                                cycle_momentum=False,
                                                                anneal_strategy='linear')

        self.criterion = Criterion(config['loss_functions']['args'])

        if cfg_trainer['resume']:
            self._resume_checkpoint(cfg_trainer['resume'])

        self.loss_scaler = torch.cuda.amp.GradScaler(enabled=cfg_trainer['use_mixed_precision'])

    def backwards(self, loss):
        self.loss_scaler.scale(loss).backward()
        self.loss_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.loss_scaler.step(self.optimizer)
        self.lr_scheduler.step()
        self.loss_scaler.update()

    def _save_checkpoint(self, epoch, iter=''):
        """
        Saving checkpoints
        :param epoch: current epoch number
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        if iter=='':
            filename = f'{self.checkpoint_dir}/last.pth'
        else:
            filename = f'{self.checkpoint_dir}/checkpoint-epoch{epoch}-iter{iter}.pth'
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    def __init__(self, config , model, data_loader):
        super().__init__(config , model, data_loader)

    def run(self):
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            FREQ = 1000
            FREQ_save = 20000
            train_log_dic = {}
            steps = 0
            for batch_idx, (data_blob, target) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                ref_img, mov_img, context_img = [x.cuda() for x in data_blob]
                flow_predictions = self.model(ref_img, mov_img, context_img)
                loss_dic = self.criterion(flow_predictions, target)
                loss = loss_dic['total_loss']
                self.backwards(loss)

                for key in loss_dic:
                    if key not in train_log_dic.keys():
                        train_log_dic[key] = 0
                    train_log_dic[key] += loss_dic[key].item()

                if steps % FREQ == FREQ - 1:
                    train_log_dic = {k: v / FREQ for k, v in train_log_dic.items()}
                    print(f'{batch_idx}/{len(self.data_loader)}', train_log_dic)
                    # print('loss', f'{batch_idx}/{len(self.data_loader)}', loss.item())
                    #
                    if not self.debug:
                        wandb.log(train_log_dic)
                    train_log_dic = {}
                if steps % FREQ_save == FREQ_save - 1:
                    self._save_checkpoint(epoch, steps)

                steps += 1

            train_log_dic = {k: v / len(self.data_loader) for k, v in train_log_dic.items()}
            # train_log_dic['lr'] = scheduler.get_last_lr()

            print('Epoch-{0} lr: {1}'.format(epoch, self.optimizer.param_groups[0]['lr']))
            print('train logs:\n', train_log_dic)

            if not self.debug:
                self._save_checkpoint(epoch)



