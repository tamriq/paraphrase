import argparse
import json
import time
import math

import torch
import mlflow.pytorch
import numpy as np
from tqdm.auto import tqdm

import transformer_model
import build_data
import residual_lstm_model


class Trainer:
    def __init__(self, model, train_loader, valid_loader, lr, pad_idx, n_epochs, device):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.n_epochs = n_epochs
        self.device = device
        # Get the optimizer.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Get the loss function.
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    def run(self):
        """"""
        best_valid_loss = float('inf')
        for epoch in range(self.n_epochs):
            start_time = time.time()
            train_loss = self.train(epoch)
            valid_loss = self.eval(epoch)
            end_time = time.time()
            epoch_mins, epoch_secs = self._count_epoch_time(start_time, end_time)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f"epoch_{epoch}.pt")
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    def train(self, epoch):
        """"""
        model.train()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Epoch {}'.format(epoch + 1))
        epoch_loss = 0
        losses_list = []
        for i, batch in enumerate(progress_bar):
            enc = batch[1][0].to(self.device)
            dec = batch[1][1].to(self.device)
            max_ind = []
            for line in enc:
                m = max([ind for ind, f in enumerate(line) if f != 0])
                max_ind.append(m)
            max_enc_len = max(max_ind) + 1
            # Crop encoder tensor according to the longest sequence.
            enc = enc[:, :max_enc_len]
            self.optimizer.zero_grad()
            output, _ = model(enc, dec[:, :-1])
            # output shape [batch size, trg len - 1, output dim]
            # trg shape [batch size, trg len]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = dec[:, 1:].contiguous().view(-1)
            # output shape [batch size * trg len - 1, output dim]
            # trg shape [batch size * trg len - 1]
            loss = self.criterion(output, trg)
            loss.backward()
            losses_list.append(float(loss))
            progress_bar.set_postfix(train_loss=np.mean(losses_list[-100:]))
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)

    def eval(self, epoch):
        """"""
        model.eval()
        losses_list = []
        epoch_loss = 0
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader),
                            desc='Epoch {}'.format(epoch + 1))
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
                output, _ = model(enc, dec[:, :-1])
                # output shape [batch size, trg len - 1, output dim]
                # trg shape [batch size, trg len]
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = dec[:, 1:].contiguous().view(-1)
                # output shape [batch size * trg len - 1, output dim]
                # trg shape [batch size * trg len - 1]
                loss = self.criterion(output, trg)
                losses_list.append(float(loss))
                progress_bar.set_postfix(test_loss=np.mean(losses_list[-100:]))
                epoch_loss += loss.item()
        return epoch_loss / len(self.valid_loader)

    @staticmethod
    def _count_epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training parameters provided at run time from the CLI'
    )
    parser.add_argument(
        '-mc',
        '--model_config',
        type=str,
        help='model parameters',
    )
    args, unknown = parser.parse_known_args()
    if args.model_config is None:
        raise argparse.ArgumentError(args.model_config, "Training parameters config file name is not specified.")
    try:
        config = json.load(open(args.model_config))
    except FileNotFoundError:
        raise
    except json.decoder.JSONDecodeError:
        raise
    # Add the current computation method to the config.
    config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the model.
    model = transformer_model.build_model(**config)
    # Get training and validation DataLoaders.
    train_loader, valid_loader = build_data.get_data_loader_train_val(**config)
    # Initialize the Trainer class.
    trainer = Trainer(model, train_loader, valid_loader, config["lr"],
                      config["pad_idx"], config["n_epochs"], config["device"])
    trainer.run()

