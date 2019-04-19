import logging


class Model:
    def __init__(self, network, loss, optimizer):
        self.network = network
        self.loss = loss
        self.optimizer = optimizer

    def train(self, dataloader, iterations, device=None, logger=None):
        # validate dataset properties
        dataset = dataloader.dataset

        assert dataset.color_space == dataset.COLOR_SPACE_LAB

        # switch to training mode (essential for batch normalization)
        self.network.train()

        # move model to device
        if device is not None:
            self.network.to(device)

        # default to dummy logger doing nothing
        logger = logger if logger is not None else logging.getLogger('dummy')

        # optimization loop
        i = 1
        while i <= iterations:
            for img in dataloader:
                # move data to device
                if device is not None:
                    img = img.to(device)

                # perform parameter update
                self.optimizer.zero_grad()

                q_pred, q_actual = self.network(img)
                loss = self.loss(q_pred, q_actual)
                loss.backward()

                self.optimizer.step()

                # display progress
                fmt = "iteration {:,}/{:,}: loss was {:1.3e}"
                msg = fmt.format(i, iterations, loss)

                logger.info(msg)

                # increment iteration counter
                i += 1

                if i > iterations:
                    break
