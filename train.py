import argparse
import json
import math
import os
import datetime
import pickle
from typing import Dict, Any

import torch
import mlflow.pytorch
import numpy as np
from tqdm.auto import tqdm

import mlflow
import mlflow.pytorch

from models import transformer
from utils import preprocessing_utils


class Trainer:
    def __init__(self, model, train_loader, valid_loader, config: Dict[str, Any], _bpe_path):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        if config["meta"]["test_run"]:
            # Add `_test` to the name of the test run.
            name = f'{config["meta"]["run_name"]}_test'
        self.name = name
        self.log: bool = config["meta"]["mlflow_log"]
        if self.log:
            self._log_start(self.name, config["meta"]["run_ver"], _bpe_path)
        self.n_epochs = config["model"]["n_epochs"]
        self.device = config["device"]
        self.clip = config["model"]["clip"]
        self.ver = config["meta"]["run_name"]
        if config["meta"]["test_run"]:
            name = f'{config["meta"]["run_name"]}_test'
        self.name = name
        # Get the optimizer.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["lr"])
        # Get the loss function.
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=config["data"]["pad_idx"])

    def run(self):
        """"""
        self._log_start_run()
        self._log_data_loaders()
        for epoch in range(self.n_epochs):
            #
            start_time = datetime.datetime.now()
            train_loss = self.train(epoch)
            valid_loss = self.eval(epoch)
            #
            timedelta = datetime.datetime.now() - start_time
            self._log_checkpoint(epoch+1, train_loss)
            # Log the metrics.
            self._log_metrics(epoch+1, timedelta, train_loss, valid_loss)
        self._log_end_run()

    def train(self, epoch: int):
        """"""
        model.train()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                            desc='Epoch {}'.format(epoch + 1))
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

    def eval(self, epoch: int):
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

    def _log_checkpoint(self, epoch, loss):
        if self.log:
            state_dict = {
                "model": model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss,
            }
            mlflow.pytorch.log_state_dict(state_dict, "checkpoint")

    def _log_metrics(self, epoch, delta, train_loss, valid_loss):
        if self.log:
            metrics = {
                "time": float(delta.seconds),
                "train_loss": float(train_loss),
                "train_ppl": math.exp(train_loss),
                "valid_loss": float(valid_loss),
                "valid_ppl": float(math.exp(valid_loss))
            }
            mlflow.log_metrics(metrics, epoch)
        print(f'Epoch: {epoch + 1:02} | Time: {delta}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    def _log_end_run(self):
        if self.log:
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.end_run()
        else:
            torch.save(self.model.state_dict(), f"model_{self.name}.pt")

    @staticmethod
    def _log_start(name, run_ver, _bpe_path):
        log_dir = "mlflow_tracking"
        mlflow.set_tracking_uri(f"file:///Users/rysshe/Documents/CODE/paraphrase/{log_dir}")
        mlflow.start_run(run_name=name)
        mlflow.log_artifact(f"configs_{run_ver}.json")
        mlflow.log_artifact(_bpe_path, "model")

    def _log_data_loaders(self):
        if self.log:
            # Log the train and validation DataLoaders.
            os.makedirs("dataloaders_tmp", exist_ok=True)
            with open("dataloaders_tmp/train_loader.pickle", 'wb') as f:
                pickle.dump(self.train_loader, f)
            with open("dataloaders_tmp/valid_loader.pickle", 'wb') as f:
                pickle.dump(self.valid_loader, f)
            mlflow.log_artifacts("dataloaders_tmp", artifact_path="dataloaders")
            os.rmdir("dataloaders_tmp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training parameters provided at run time from the CLI'
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        help='Learning parameters for the dataset and the model',
    )
    args, unknown = parser.parse_known_args()
    if args.config is None:
        raise argparse.ArgumentError(args.config, "Training parameters config file name is not specified.")
    try:
        config = json.load(open(args.config))
    except FileNotFoundError:
        raise
    except json.decoder.JSONDecodeError:
        raise
    # Add the current computation method to the config.
    config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the model.
    model = transformer.build_model(**config)
    print("Model is initialized.")
    # Get training and validation DataLoaders.
    (train_loader, valid_loader), _bpe_path = preprocessing_utils.build_data(**config)
    print("DataLoaders are built.")
    # Initialize the Trainer class.
    trainer = Trainer(model, train_loader, valid_loader, config, _bpe_path)
    print("Trainer is Initialized.")
    trainer.run()

