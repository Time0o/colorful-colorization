import logging
import os
from glob import glob

import torch


# TODO: save/restore across devices

class Model:
    CHECKPOINT_PREFIX = 'checkpoint'
    CHECKPOINT_POSTFIX = 'tar'
    CHECKPOINT_ID_FMT = '{:010}'

    def __init__(self,
                 network,
                 loss,
                 optimizer,
                 device=None,
                 checkpoint_dir=None,
                 logger=None):

        if checkpoint_dir is not None and not os.path.isdir(checkpoint_dir):
            raise ValueError("checkpoint directory must exist")

        self.network = network
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        if logger is not None:
            if isinstance(logger, str):
                self.logger = logging.getLogger(logger)
            else:
                self.logger = logger
        else:
            self.logger = logging.getLogger('dummy')

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, d):
        self._device = d

        # move network to device
        if d is not None:
            self.network.to(d)

    def train(self,
              dataloader,
              iterations,
              epoch_init=None,
              epochs_till_checkpoint=None):

        # refuse to overwrite checkpoints unless resuming
        if epoch_init is None:
            checkpoint_files = os.listdir(self.checkpoint_dir)

            if len([f for f in checkpoint_files if not f.startswith('.')]) > 0:
                raise ValueError("checkpoint directory must be empty")

        # switch to training mode (essential for batch normalization)
        self.network.train()

        # optimization loop
        if epoch_init is None:
            i = 1
            ep = 1
        else:
            i = len(dataloader) * (epoch_init - 1) + 1
            ep = epoch_init

        while i <= iterations:
            for img in dataloader:
                # move data to device
                if self.device is not None:
                    img = img.to(self.device)

                # perform parameter update
                self.optimizer.zero_grad()

                q_pred, q_actual = self.network(img)
                loss = self.loss(q_pred, q_actual)
                loss.backward()

                self.optimizer.step()

                # display progress
                fmt = "iteration {:,}/{:,}: loss was {:1.3e}"
                msg = fmt.format(i, iterations, loss)

                self.logger.info(msg)

                # increment iteration counter
                i += 1

                if i > iterations:
                    break

            # periodically save checkpoint
            if ep % epochs_till_checkpoint == 0:
                self.checkpoint_training(iteration=i, epoch=ep)

            ep += 1

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path)['network'])

    def checkpoint_training(self, iteration, epoch):
        state = {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iteration': iteration,
            'epoch': epoch
        }

        path = self._checkpoint_path(epoch)

        torch.save(state, path)

        self.logger.info("saved checkpoint '{}'".format(os.path.basename(path)))

    def resume_training(self,
                        dataloader,
                        iterations,
                        epochs_till_checkpoint=None,
                        checkpoint_iteration=None):

        # load checkpoint
        if checkpoint_iteration is not None:
            path = self._checkpoint_path(checkpoint_iteration)

            if not os.path.exists(path):
                raise ValueError("failed to find checkpoint '{}'".format(path))
        else:
            checkpoint_template = '{}_*.{}'.format(self.CHECKPOINT_PREFIX,
                                                   self.CHECKPOINT_POSTFIX)

            checkpoint_template = os.path.join(self.checkpoint_dir,
                                               checkpoint_template)

            all_checkpoints = sorted(glob(checkpoint_template))

            if not all_checkpoints:
                err = "failed to resume training: no previous checkpoints"
                raise ValueError(err)

            path = all_checkpoints[-1]

        state = torch.load(path)

        # load network weights
        self.network.load_state_dict(state['network'])

        # load optimizer state
        self.optimizer.load_state_dict(state['optimizer'])

        # resume training
        self.train(dataloader,
                   iterations,
                   epoch_init=state['epoch'],
                   epochs_till_checkpoint=epochs_till_checkpoint)

    def _checkpoint_path(self, epoch):
        checkpoint = '{}_{}.{}'.format(self.CHECKPOINT_PREFIX,
                                       self.CHECKPOINT_ID_FMT.format(epoch),
                                       self.CHECKPOINT_POSTFIX)

        return os.path.join(self.checkpoint_dir, checkpoint)
