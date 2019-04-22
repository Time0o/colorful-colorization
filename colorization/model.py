import logging
import os
import re
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
                 logger=None):

        self.network = network
        self.loss = loss
        self.optimizer = optimizer
        self.device = device

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
              checkpoint_dir=None,
              epochs_till_checkpoint=None):

        # validate checkpoint directory
        if checkpoint_dir is not None:
            self._validate_checkpoint_dir(checkpoint_dir,
                                          resuming=(epoch_init is not None))

        # switch to training mode (essential for batch normalization)
        self.network.train()

        # optimization loop
        if epoch_init is None:
            i = 1
            ep = 1
        else:
            i = len(dataloader) * (epoch_init - 1) + 1
            ep = epoch_init


        done = False
        while not done:
            for img, _ in dataloader:
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
                    done = True
                    break

            # periodically save checkpoint
            if not done:
                if checkpoint_dir is not None:
                    if ep % epochs_till_checkpoint == 0:
                        self.checkpoint_training(checkpoint_dir, ep)

                ep += 1

        # save final checkpoint
        if checkpoint_dir is not None:
            self.checkpoint_training(checkpoint_dir, 'final')

    def predict(self, img):
        # switch to evaluation mode
        self.network.eval()

        # move data to device
        if self.device is not None:
            img_device = img.to(self.device)

        # run prediction
        with torch.no_grad():
            img_pred_device = self.network(img_device)
            img_pred = img_pred_device.to(img.device)

        return img_pred

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path)['network'])

    def checkpoint_training(self, checkpoint_dir, checkpoint_epoch):
        state = {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        path = self._checkpoint_path(checkpoint_dir, checkpoint_epoch)

        torch.save(state, path)

        self.logger.info("saved checkpoint '{}'".format(os.path.basename(path)))

    def restore_training(self, checkpoint_dir, checkpoint_epoch=None):
        # load checkpoint
        if checkpoint_epoch is not None:
            checkpoint_path = self._checkpoint_path(checkpoint_dir,
                                                    checkpoint_epoch)

            if not os.path.exists(path):
                raise ValueError("failed to find checkpoint '{}'".format(path))
        else:
            # skip the final checkpoint here because it might not correspond to
            # the end of an epoch
            checkpoint_path, checkpoint_epoch = \
                self.find_latest_checkpoint(checkpoint_dir, skip_final=True)

        # load checkpoint
        state = torch.load(checkpoint_path)

        # load network weights
        self.network.load_state_dict(state['network'])

        # load optimizer state
        self.optimizer.load_state_dict(state['optimizer'])

        # return checkpoint epoch
        return checkpoint_epoch

    @classmethod
    def find_latest_checkpoint(cls, checkpoint_dir, skip_final=False):
        # create list of all checkpoints
        checkpoint_template = '{}_*.{}'.format(cls.CHECKPOINT_PREFIX,
                                               cls.CHECKPOINT_POSTFIX)

        checkpoint_template = os.path.join(checkpoint_dir,
                                           checkpoint_template)

        all_checkpoints = sorted(glob(checkpoint_template))

        # find lastest checkpoint
        while True:
            if not all_checkpoints:
                err = "failed to resume training: no previous checkpoints"
                raise ValueError(err)

            checkpoint_path = all_checkpoints[-1]

            is_final = checkpoint_path.find('final') != -1

            if is_final and skip_final:
                all_checkpoints.pop()
            else:
                break

        # deduce checkpoint epoch from filename
        if is_final:
            return checkpoint_path, 'final'
        else:
            checkpoint_regex = checkpoint_template.replace('*', '(\\d+)')

            m = re.match(checkpoint_regex, checkpoint_path)

            checkpoint_epoch = int(m.group(1))

            return checkpoint_path, checkpoint_epoch

    @staticmethod
    def _validate_checkpoint_dir(checkpoint_dir, resuming=False):
        # check existance
        if not os.path.isdir(checkpoint_dir):
            raise ValueError("checkpoint directory must exist")

        # refuse to overwrite checkpoints unless resuming
        if not resuming:
            checkpoint_files = os.listdir(checkpoint_dir)

            if len([f for f in checkpoint_files if not f.startswith('.')]) > 0:
                raise ValueError("checkpoint directory must be empty")

    @classmethod
    def _checkpoint_path(cls, checkpoint_dir, checkpoint_epoch):
        if checkpoint_epoch == 'final':
            checkpoint_id = 'final'
        else:
            checkpoint_id = cls.CHECKPOINT_ID_FMT.format(checkpoint_epoch)

        checkpoint = '{}_{}.{}'.format(cls.CHECKPOINT_PREFIX,
                                       checkpoint_id,
                                       cls.CHECKPOINT_POSTFIX)

        return os.path.join(checkpoint_dir, checkpoint)
