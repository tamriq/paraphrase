import math
import datetime

import torch
import numpy as np
from tqdm.auto import tqdm

from models import transformer
import utils.preprocessing as putils


class Trainer:
    def __init__(self, model, train_loader, valid_loader, n_epochs, device, clip, lr, pad):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.n_epochs = n_epochs
        self.device = device
        self.clip = clip
        # Get the optimizer.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Get the loss function.
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad)

    def train(self, epoch: int):
        """
        Train epoch.

        :param epoch:
        :return:
        """
        self.model.train()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                            desc='Epoch {}'.format(epoch))
        epoch_loss = 0
        losses_list = []
        for i, batch in enumerate(progress_bar):
            enc = batch[1][0].to(self.device)
            dec = batch[1][1].to(self.device)
            max_ind = []
            for line in enc:
                # Get the actual length of the encoder sequence before the padding.
                m = max([ind for ind, f in enumerate(line) if f != 0])
                max_ind.append(m)
            # Get the maximal sequence length in a batch.
            max_enc_len = max(max_ind) + 1
            # Crop encoder tensor according to the longest sequence in a batch.
            enc = enc[:, :max_enc_len]
            self.optimizer.zero_grad()
            # Process the pair.
            output, _ = self.model(enc, dec[:, :-1])
            output_dim = output.shape[-1]
            # Reshape output tensor to calculate the loss.
            # [batch size, trg len - 1, output dim] -> [batch size * trg len - 1, output dim]
            output = output.contiguous().view(-1, output_dim)
            # Reshape decoder tensor to calculate the loss.
            # [batch size, trg len] -> [batch size * trg len - 1]
            dec = dec[:, 1:].contiguous().view(-1)
            # Calculate the loss.
            loss = self.criterion(output, dec)
            #
            loss.backward()
            losses_list.append(float(loss))
            progress_bar.set_postfix(train_loss=np.mean(losses_list[-100:]))
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)

    def eval(self, epoch: int):
        """
        Evaluate epoch.

        :param epoch:
        :return:
        """
        self.model.eval()
        losses_list = []
        epoch_loss = 0
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader),
                            desc='Epoch {}'.format(epoch))
        with torch.no_grad():
            for i, batch in enumerate(progress_bar):
                enc = batch[1][0].to(self.device)
                dec = batch[1][1].to(self.device)
                max_ind = []
                for line in enc:
                    m = max([ind for ind, f in enumerate(line) if f != 0])
                    max_ind.append(m)
                max_enc_len = max(max_ind) + 1
                enc = enc[:, :max_enc_len]  #
                output, _ = self.model(enc, dec[:, :-1])
                # Reshape output tensor to calculate the loss.
                # [batch size, trg len - 1, output dim] -> [batch size * trg len - 1, output dim]
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                # Reshape decoder tensor to calculate the loss.
                # [batch size, trg len] -> [batch size * trg len - 1]
                dec = dec[:, 1:].contiguous().view(-1)
                # Calculate the loss.
                loss = self.criterion(output, dec)
                losses_list.append(float(loss))
                progress_bar.set_postfix(test_loss=np.mean(losses_list[-100:]))
                epoch_loss += loss.item()
        return epoch_loss / len(self.valid_loader)


class TrainHelper(Trainer):
    def __init__(self, configs):
        model = transformer.build_model(configs)
        train_loader, valid_loader, _bpe_path = putils.build_data(configs)
        self._bpe_path = _bpe_path
        self.configs = configs
        self.training_log = {}
        super().__init__(model,
                         train_loader,
                         valid_loader,
                         configs["model"]["n_epochs"],
                         configs["device"],
                         configs["model"]["clip"],
                         configs["model"]["lr"],
                         configs["data"]["pad_idx"])

    def run(self):
        """


        :return:
        """
        for epoch in range(self.n_epochs):
            epoch += 1
            log = self.run_epoch(epoch)
            self._print_metrics(epoch, log)
        torch.save(self.model.state_dict(), 'trained-model.pt')

    def run_epoch(self, epoch: int):
        epoch_log = {}
        start_time = datetime.datetime.now()
        epoch_log["train_loss"] = self.train(epoch)
        epoch_log["valid_loss"] = self.eval(epoch)
        # Get the duration of the epoch.
        epoch_log["duration"] = datetime.datetime.now() - start_time
        # Save the log.
        self.training_log[epoch] = epoch_log
        return epoch_log

    @staticmethod
    def _print_metrics(epoch, log):
        print(f'Epoch: {epoch:02} | Time: {log["duration"]}')
        print(f'\tTrain Loss: {log["train_loss"]:.3f} | Train PPL: {math.exp(log["train_loss"]):7.3f}')
        print(f'\t Val. Loss: {log["valid_loss"]:.3f} |  Val. PPL: {math.exp(log["valid_loss"]):7.3f}')


