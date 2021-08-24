import pickle
import os
import math
import shutil

import mlflow
import mlflow.pytorch
import utils.training as tutils


class MLflowWrapper(tutils.Trainer):
    """
    MLflow logging for Trainer class.
    """

    def __init__(self, configs):
        # Inherit from Trainer.
        super().__init__(configs)
        self._start_mlflow(self.configs["meta"]["run_ver"],
                           self.configs["meta"]["mlflow_dir"])
        self._log_training_data_artifacts(self.configs["meta"]["run_ver"],
                                          self._bpe_path,
                                          self.train_loader,
                                          self.valid_loader)

    def run(self):
        """
        Run the training loop and log the training data MLflow.
        """
        for epoch in range(self.n_epochs):
            epoch += 1
            log = self.run_epoch(epoch)
            self._print_metrics(epoch, log)
            print(f"Epoch {epoch}")
            self._log_metrics(
                epoch,
                log["duration"],
                log["train_loss"],
                log["valid_loss"]
            )
        # Save the checkpoint from the last iteration.s
        self._log_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            log["train_loss"]
        )
        # Save the model.
        mlflow.pytorch.log_model(self.model, "model")
        mlflow.end_run()

    @staticmethod
    def _log_checkpoint(model, optimizer, epoch, loss):
        """
        Save the checkpoint.

        :param model: the model instance to log
        :param optimizer: the optimizer instance to log
        :param epoch: the current number of epoch
        :param loss: the current loss
        :return:
        """
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }
        mlflow.pytorch.log_state_dict(state_dict, "checkpoint")

    @staticmethod
    def _log_metrics(epoch, delta, train_loss, eval_loss):
        """
        Log epoch metrics.

        :param epoch: the current number of epoch
        :param delta: the duration of epoch
        :param train_loss: train loss
        :param eval_loss: evaluation loss
        :return:
        """
        metrics = {
            "duration": float(delta.seconds),
            "train_loss": float(train_loss),
            "train_ppl": math.exp(train_loss),
            "eval_loss": float(eval_loss),
            "eval_ppl": float(math.exp(eval_loss))
        }
        mlflow.log_metrics(metrics, epoch)

    @staticmethod
    def _start_mlflow(run_name: str, mlflow_log_dir: str):
        """
        Start MLflow session.

        :param run_name: the version of MLflow run, e.g. `v0.2_test`
        :param mlflow_log_dir: the name of mlflow project directory
        :return:
        """
        mlflow_abs = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), mlflow_log_dir)
        tracking_uri = f"file://{mlflow_abs}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.start_run(run_name=run_name)

    def _log_training_data_artifacts(self, run_name, _bpe_path, train_loader, valid_loader):
        """
        Log the data that was prepared for the training to make the process reproducible.

        :param run_name: the version of MLflow run, e.g. `v0.2_test`
        :param _bpe_path: the path to BPE model that was trained on the data
        :param train_loader: training data generator
        :param valid_loader: evaluation data generator
        :return:
        """
        # Log config file.
        mlflow.log_artifact(f"configs/{run_name}.json", "preprocessing")
        mlflow.log_artifact(_bpe_path, "preprocessing")
        os.remove(_bpe_path)
        # Log the train and validation DataLoaders.
        os.makedirs("dataloaders_tmp", exist_ok=True)
        with open("dataloaders_tmp/train_loader.pickle", 'wb') as f:
            pickle.dump(train_loader, f)
        with open("dataloaders_tmp/valid_loader.pickle", 'wb') as f:
            pickle.dump(valid_loader, f)
        mlflow.log_artifacts("dataloaders_tmp", "preprocessing")
        shutil.rmtree("dataloaders_tmp")
