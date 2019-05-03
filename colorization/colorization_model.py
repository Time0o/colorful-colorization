import logging
import logging.config
import os
import re
from glob import glob
from math import exp, log
from warnings import warn

import torch
from torch.multiprocessing import Process, SimpleQueue, set_start_method
from torch.optim.lr_scheduler import LambdaLR, CyclicLR

try:
    set_start_method('spawn', force=True)
    _mp_spawn = True
except RuntimeError:
    warn("failed to set start method to 'spawn', logging will be disabled")
    _mp_spawn = False


class _IterationLogData:
    def __init__(self, i, i_max, lr=None, loss=None):
        self.i = i
        self.i_max = i_max
        self.lr = lr
        self.loss = loss

    def __repr__(self):
        msg = "iteration {:,}/{:,}".format(self.i, self.i_max)

        if self.lr is not None:
            msg += ", lr was {:1.3e}".format(self.lr)

        if self.loss is not None:
            msg += ", loss was {:1.3e}".format(self.loss)

        return msg


def _log_progress(log_config, logger, queue):
    logging.config.dictConfig(log_config)
    logger = logging.getLogger(logger)

    while True:
        data = queue.get()

        if data == 'done':
            break

        logger.info(data)


class ColorizationModel:
    """ Top-level wrapper class implementing training and prediction.

    This is a wrapper class that composes a PyTorch network with a loss
    function and an optimizer and implements training, prediction,
    checkpointing and logging functionality. Its implementation is kept
    independent of the concrete underlying network.

    """

    CHECKPOINT_PREFIX = 'checkpoint'
    CHECKPOINT_POSTFIX = 'tar'
    CHECKPOINT_ID_FMT = '{:010}'

    def __init__(self,
                 network,
                 loss=None,
                 optimizer=None,
                 lr_scheduler=None,
                 log_config=None,
                 logger=None):
        """
        Compose a model.

        Note:
            The model is not trainable unless both `loss` and `optimizer` are
            not `None`. A non-trainable model can still be initialized from
            pretrained weights for evaluation or prediction purposes.

            Likewise, logging will only be enabled if both `log_config` and
            `logger` are not `None`.

        Args:
            network (colorization.modules.ColorizationNetwork):
                Network instance.
            loss (torch.nn.Module, optional):
                Training loss function, if this is set to `None`, the model is
                not trainable.
            optimizer (functools.partial, optional):
                Partially applied training optimizer, parameter argument is
                supplied by this constructor, if this is set to `None`, the
                model is not trainable.
            log_config (dict, optional):
                Python `logging` configuration dictionary, if this is set to
                `None`, logging will be disabled.
            logger (str, optional):
                Name of the logger to be utilized (this logger has to be
                configured by `log_config`, pre-existing loggers will not
                function correctly since logging is performed in a separate
                thread).

        """

        self.network = network
        self.loss = loss
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

        self._log_enabled = \
            _mp_spawn and log_config is not None and logger is not None

        if self._log_enabled:
            self._log_config = log_config
            self._logger = logger

    def lr_find(self,
                dataloader,
                lr_min=1e-7,
                lr_max=0.1,
                lr_exp=False,
                iterations=100,
                smoothing_factor=0.05):
        """Utility function for finding good cyclical learning rate limits.

        This function increases the learning rate over a certain number of
        iterations and records the (smoothed) resulting losses.

        Args:
            dataloader (torch.utils.data.DataLoder):
                Data loader, should return Lab images of arbitrary shape and
                datatype float32. For efficient training, the data loader
                should be constructed with `pin_memory` set to `True`.
            lr_min (float):
                Initial learning rate.
            lr_max (float):
                Final learning rate.
            lr_exp (bool):
                If `True`, increase learning rate exponentially instead of
                linearly.
            iterations (int):
                Number of iterations (batches) to run the training for.
            smoothing_factor:
                Loss smoothing factor, should be between 0 and 1, lower
                values result in smoother loss curves.

        """

        # start training
        self._before_training()

        # construct optimizer
        self._create_optimizer(create_scheduler=False)

        self._set_lr(lr_min)

        # construct scheduler
        if lr_exp:
            def lr_lambda(i):
                if i == 0:
                    return 1

                return exp((log(lr_max) - log(lr_min)) * (i / (iterations - 1)))
        else:
            def lr_lambda(i):
                if i == 0:
                    return 1

                return (lr_max / lr_min) * (i / (iterations - 1))

        lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

        # create loss curve
        learning_rates = []
        losses = []

        i = 1
        done = False
        while not done:
            for img in dataloader:
                # move image to device
                img = self._to_device(img, dataloader)

                # update parameters
                loss = self._update_parameters(img)

                # store loss
                if i > 1:
                    loss = smoothing_factor * loss + \
                           (1 - smoothing_factor) * losses[-1]

                losses.append(loss)

                # display progress
                lr = self._get_lr()

                if self._log_enabled:
                    log_data = _IterationLogData(i, iterations, lr=lr)

                    self._log_queue.put(log_data)

                # update learning rate
                learning_rates.append(lr)

                lr_scheduler.step()

                # increment iteration counter
                i += 1

                if i > iterations:
                    done = True
                    break

        # end training
        self._after_training()

        return learning_rates, losses

    def train(self,
              dataloader,
              iterations,
              checkpoint_init=None,
              checkpoint_dir=None,
              epochs_till_checkpoint=None):
        """Train the models network.

        Args:
            dataloader (torch.utils.data.DataLoder):
                Data loader, should return Lab images of arbitrary shape and
                datatype float32. For efficient training, the data loader
                should be constructed with `pin_memory` set to `True`.
            iterations (int):
                Number of iterations (batches) to run the training for, note
                that training can be picked up at a previous checkpoint later
                on, also note that while training time is specified in
                iterations, checkpoint frequency is specified in epochs to
                avoid issues with data loader shuffling etc.
            checkpoint_init (str, optional):
                Previous checkpoint from which to pick up training.
            checkpoint_dir (str, optional):
                Directory in which to save checkpoints, must exist and be
                empty, is this is set to `None`, no checkpoints will be saved.
            epochs_till_checkpoint (str, optional):
                Number of epochs between checkpoints, only meaningful in
                combination with `checkpoint_dir`.

        """

        # restore from checkpoint
        if checkpoint_init is not None:
            self._load_checkpoint(checkpoint_init, load_optimizer=True)
            epoch_init = self._checkpoint_epoch(checkpoint_init)

        # validate checkpoint directory
        if checkpoint_dir is not None:
            self._validate_checkpoint_dir(
                checkpoint_dir, resuming=(checkpoint_init is not None))

        # start training
        self._before_training()

        # construct optimizer and scheduler
        self._create_optimizer()

        # optimization loop
        if checkpoint_init is None:
            i = 1
            ep = 1
        else:
            i = len(dataloader) * (epoch_init - 1) + 1
            ep = epoch_init

        done = False
        while not done:
            for img in dataloader:
                # move image to device
                img = self._to_device(img, dataloader)

                # update parameters
                loss = self._update_parameters(img)

                # update learning rate
                self.lr_scheduler.step()

                # display progress
                lr = self._get_lr()

                if self._log_enabled:
                    log_data = _IterationLogData(
                        i, iterations, loss=loss.item(), lr=lr)

                    self._log_queue.put(log_data)

                # increment iteration counter
                i += 1

                if i > iterations:
                    done = True
                    break

            # periodically save checkpoint
            if not done:
                if checkpoint_dir is not None:
                    if ep % epochs_till_checkpoint == 0:
                        self._checkpoint_training(checkpoint_dir, ep)

                ep += 1

        # save final checkpoint
        if checkpoint_dir is not None:
            self._checkpoint_training(checkpoint_dir, 'final')

        # end training
        self._after_training()

    def predict(self, img):
        """Perform single batch prediction using the current network.

        Args:
            img (torch.Tensor):
                A tensor of shape `(n, 1, h, w)` where `n` is the size of the
                batch to be predicted and `h` and `w` are image dimensions. The
                image should be the lightness channel of an image converted into
                the Lab color space.

        Returns:
            A tensor of shape `(n, 1, h, w)` containing the predicted ab
            channels.

        """
        # switch to evaluation mode
        self.network.eval()

        # move data to device
        img = img.cuda()

        # run prediction
        with torch.no_grad():
            img_pred = self.network(img)

        return img_pred

    def restore(self, checkpoint_path):
        """Initialize model weights from a training checkpoint.

        Args:
            checkpoint_path (str):
                Path to the training checkpoint.

        """

        self._load_checkpoint(checkpoint_path)

    @classmethod
    def find_latest_checkpoint(cls, checkpoint_dir, skip_final=False):
        """Find the most up to date checkpoint file in a checkpoint directory.

        Args:
            checkpoint_dir (str):
                Directory in which to search for checkpoints, must exist.
            skip_final (bool):
                If `True` don't consider the checkpoint created after the final
                training iteration. This is sensible if the returned checkpoint
                will be be used to resume training since training can only be
                resumed from checkpoints created at the end of an epoch.

        Returns:
            The file path to the latest checkpoint as well as the epoch at which
            that checkpoint was created in a tuple.

        """

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

        return checkpoint_path

    def _create_optimizer(self, create_scheduler=True):
        self.optimizer = self._optimizer(self.network.parameters())

        if self._lr_scheduler is None:
            class DummyScheduler:
                def step(self):
                    pass

            self.lr_scheduler = DummyScheduler()
        else:
            # circumvent PyTorch CyclicLR bug
            for pg in self.optimizer.param_groups:
                if 'momentum' not in pg:
                    pg['momentum'] = None

            self.lr_scheduler = self._lr_scheduler(self.optimizer)

    def _get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def _set_lr(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr

    def _before_training(self):
        # switch to training mode (essential for batch normalization)
        self.network.train()

        # create logging thread
        if self._log_enabled:
            self._log_queue = SimpleQueue()

            self._log = Process(
                target=_log_progress,
                args=(self._log_config, self._logger, self._log_queue))

            self._log.start()

    def _after_training(self):
        # stop logging thread
        if self._log_enabled:
            self._log_queue.put('done')
            self._log.join()

    def _to_device(self, img, dataloader):
        # move data to device
        if dataloader.pin_memory:
            return img.cuda(non_blocking=True)
        else:
            warn("'pin_memory' not set, this will slow down training")
            return img.cuda()

    def _update_parameters(self, img):
        self.optimizer.zero_grad()

        q_pred, q_actual = self.network(img)
        loss = self.loss(q_pred, q_actual)
        loss.backward()

        self.optimizer.step()

        return loss

    def _checkpoint_training(self, checkpoint_dir, checkpoint_epoch):
        state = {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        path = self._checkpoint_path(checkpoint_dir, checkpoint_epoch)

        torch.save(state, path)

        if self._log_enabled:
            fmt = "saved checkpoint '{}'"
            self._log_queue.put(fmt.format(os.path.basename(path)))

    def _load_checkpoint(self, checkpoint_path, load_optimizer=False):
        state =  torch.load(checkpoint_path)

        self.network.load_state_dict(state['network'])

        if load_optimizer:
            self.optimizer.load_state_dict(state['optimizer'])

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

    @classmethod
    def _checkpoint_epoch(cls, checkpoint_path):
        if checkpoint_path.find('final') != -1:
            return 'final'

        checkpoint_regex = '{}_(\\d+).{}'.format(cls.CHECKPOINT_PREFIX,
                                                 cls.CHECKPOINT_POSTFIX)

        m = re.match(checkpoint_regex, checkpoint_path)

        checkpoint_epoch = int(m.group(1))

        return checkpoint_epoch
