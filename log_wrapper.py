import pickle
import os
import math
import shutil

import mlflow
import mlflow.pytorch
import utils.training as tutils


class MLflowWrapper(tutils.TrainHelper):
    def __init__(self, configs):
        super().__init__(configs)
        self._start_mlflow(self.configs["meta"]["run_ver"],
                           self.configs["meta"]["mlflow_dir"])
        self._log_preproc_data(self.configs["meta"]["run_ver"],
                               self._bpe_path,
                               self.train_loader,
                               self.valid_loader)

    def run(self):
        """
        """
        for epoch in range(self.n_epochs):
            epoch += 1
            log = self.run_epoch(epoch)
            self._print_metrics(epoch, log)
            print(f"Epoch {epoch}")
            self._log_metrics(epoch,
                              log["duration"],
                              log["train_loss"],
                              log["valid_loss"])
        self._log_checkpoint(self.model, self.optimizer, epoch, log["train_loss"])
        # Save the model.
        mlflow.pytorch.log_model(self.model, "model")
        mlflow.end_run()

    @staticmethod
    def _log_checkpoint(model, optimizer, epoch, loss):
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }
        mlflow.pytorch.log_state_dict(state_dict, "checkpoint")

    @staticmethod
    def _log_metrics(epoch, delta, train_loss, valid_loss):
        metrics = {
            "time": float(delta.seconds),
            "train_loss": float(train_loss),
            "train_ppl": math.exp(train_loss),
            "valid_loss": float(valid_loss),
            "valid_ppl": float(math.exp(valid_loss))
        }
        mlflow.log_metrics(metrics, epoch)

    @staticmethod
    def _start_mlflow(run_name: str, mlflow_log_dir: str):
        """
        Start MLflow instance.

        :param run_name: the name of run, e.g. `v0.2_test`
        :param mlflow_log_dir: the name of mlflow project directory
        :return:
        """
        mlflow_abs = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), mlflow_log_dir)
        tracking_uri = f"file://{mlflow_abs}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.start_run(run_name=run_name)

    def _log_preproc_data(self, run_ver, _bpe_path, train_loader, valid_loader):
        """
        Log all preprocessing results.

        :param run_ver:
        :param _bpe_path:
        :param train_loader:
        :param valid_loader:
        :return:
        """
        # Log config file.
        mlflow.log_artifact(f"configs/{run_ver}.json", "preprocessing")
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
