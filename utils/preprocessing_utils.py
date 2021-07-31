import math
import random
import os
from typing import Any, Dict

import torch
import pandas as pd
import youtokentome as yttm
from tqdm.auto import tqdm


class SequenceBucketingData(torch.utils.data.Dataset):
    """
    Apply sequence bucketing algorithm to the data and pack it into the pytorch data loader.
    """

    def __init__(self, data, max_len, pad_index, eos_index, bos_index):
        self.data = data
        self.max_len = max_len
        self.pad_index = pad_index
        self.eos_index = eos_index
        self.bos_index = bos_index

    def __len__(self):
        return len(self.data)

    def prepare_sample(self, sequence, max_len_dec):
        """
        Process the source-target pair of the tokens.

        :param sequence: source target pair of the tokens
        :param max_len_dec: the longest target length in the dataset
        :return: processed source and target sequences
        """
        enc = sequence[0][:self.max_len]
        dec = sequence[1][:max_len_dec]
        # Add BOS and EOS tokens to the encoder and decoder sequences.
        enc = [self.bos_index] + enc + [self.eos_index]
        dec = [self.bos_index] + dec + [self.eos_index]
        # Pad the encoder and decoder sequences.
        pads_enc = [self.pad_index] * (self.max_len + 2 - len(enc))
        pads_dec = [self.pad_index] * (max_len_dec + 2 - len(dec))
        enc += pads_enc
        dec += pads_dec
        return enc, dec

    def __getitem__(self, index):
        batch = self.data[index]
        # Find the length of the biggest target sequence.
        max_len_dec = min([self.max_len, max([len(sample[1]) for sample in batch])])
        batch_x = []
        batch_y = []
        for sample in batch:
            x, y = self.prepare_sample(sample, max_len_dec)
            batch_x.append(x)
            batch_y.append(y)
        batch_x = torch.tensor(batch_x).long()
        batch_y = torch.tensor(batch_y).long()
        return batch_x, batch_y


class Tokenizer:
    """
    Trains the BPE tokenizer and tokenizes the data.
    """

    def __init__(self, data: pd.DataFrame, vocab_size: int, run_ver: str):
        """

        :param dataset_name:
        :param vocab_size:
        """
        self.data = data
        bpe_file_name = self.write_bpe_txt(data, run_ver)
        self.bpe, self.bpe_path = self.train(run_ver, bpe_file_name, vocab_size)

    def run(self, bpe_batch_size: int = 256, shuffle: bool = True):
        """

        :param shuffle:
        :param bpe_batch_size:
        :return:
        """
        tokenized_src = []
        tokenized_trg = []
        # Iterate over the dataframe and tokenize the pairs.
        for i_batch in tqdm(range(math.ceil(len(self.data) / bpe_batch_size))):
            # Tokenize source and target sentences of the bpe-batch separately.
            src_batch = self.bpe.encode(list(self.data[0][i_batch * bpe_batch_size:(i_batch + 1) * bpe_batch_size]))
            trg_batch = self.bpe.encode(list(self.data[1][i_batch * bpe_batch_size:(i_batch + 1) * bpe_batch_size]))
            # Add the sentences tokens to the storage list.
            tokenized_src.extend(src_batch)
            tokenized_trg.extend(trg_batch)
            # Store the source and target sentences tokens in reverse order to learn all possible combinations.
            tokenized_src.extend(trg_batch)
            tokenized_trg.extend(src_batch)
        # Combine back into pairs.
        tokenized_pairs = [i for i in zip(tokenized_src, tokenized_trg)]
        if shuffle:
            random.shuffle(tokenized_pairs)
        return tokenized_pairs

    @staticmethod
    def train(run_ver: str, bpe_file_name: str, vocab_size: int):
        """

        :param run_ver:
        :param bpe_file_name:
        :param vocab_size:
        :return:
        """
        model_path = f"bpe_{run_ver}.model"
        yttm.BPE.train(data=bpe_file_name, vocab_size=vocab_size, model=model_path)
        bpe = yttm.BPE(model=model_path)
        os.remove(bpe_file_name)
        return bpe, model_path

    @staticmethod
    def write_bpe_txt(data: pd.DataFrame, data_name: str) -> str:
        """

        :param data: data in dataframe format
        :param data_name: the filename of the dataset
        :return:
        """
        err = 0
        bpe_file_name = f"for_bpe_{data_name}.txt"
        f = open(bpe_file_name, 'w')
        for src in data[0]:
            try:
                f.write(src + '\n')
            except:
                err += 1
        for trg in data[1]:
            try:
                f.write(trg + '\n')
            except:
                err += 1
        f.close()
        # We need to log here?
        print(err)
        return bpe_file_name


class Loader:
    def __init__(self, data_config: Dict[str, Any], run_ver: str, test_run: bool):
        self.data_config = data_config
        data = self.load_data(data_config["dataset_name"], data_config["dataset_dir"], test_run)
        self.tokenizer = Tokenizer(data, data_config["vocab_size"], run_ver)

    def run(self):
        tokenized_sequence = self.tokenizer.run()
        batches = self.batch_sequence(tokenized_sequence, self.data_config["batch_size"],
                                      self.data_config["sequence_bucketing"])
        random.shuffle(batches)
        # Compute the start index for validation dataset.
        split_index = int(len(batches) * self.data_config["train_val_proportion"])
        # Wrap pytorch DataLoaders around the batches.
        train_loader = self._init_loader(batches[:-split_index])
        validation_loader = self._init_loader(batches[-split_index:])
        return train_loader, validation_loader

    def _init_loader(self, batch_split):
        return SequenceBucketingData(batch_split, self.data_config["max_len"],
                                     self.data_config["pad_idx"],
                                     self.data_config["eos_idx"],
                                     self.data_config["bos_idx"])

    @staticmethod
    def load_data(name: str, dir_path: str, test_run: bool) -> pd.DataFrame:
        """
        Load the data.

        :param test_run:
        :param name: the name of the dataset
            - 'backed' for the corpus of the backtranslated pairs
            - 'news' for the corpus of the paired news titles
            - 'subtitles' for the paraphrase corpus of subtitles
        :param dir_path:
        :return: data in DataFrame format
        """
        name = name.strip()
        # assert name in ["backed", "news", "subtitles"], "Invalid data name"
        tsv_data_path = os.path.join(dir_path, f"{name}.tsv")
        df = pd.read_csv(tsv_data_path, sep='\t', header=None, error_bad_lines=False)
        # Drop Nans.
        df = df.dropna()
        # Shuffle data.
        df = df.sample(frac=1).reset_index(drop=True)
        if test_run:
            df = df[:int(df.shape[0] * 0.001)]
        return df

    @staticmethod
    def batch_sequence(sequence, batch_size: int, sort_by_src: bool):
        """

        :param sequence:
        :param batch_size:
        :param sort_by_src:
        :return:
        """
        if sort_by_src:
            # Sort the pairs by the length of the source sentence.
            # This type of sorting is required to implement the SequenceBucketing algorithm.
            sequence = sorted(sequence, key=lambda x: len(x[0]), reverse=True)
        sequence_batches = []
        for i_batch in range(math.ceil(len(sequence) / batch_size)):
            sequence_batches.append(sequence[i_batch * batch_size:(i_batch + 1) * batch_size])
        # LOG LENGTH OF BATCHES
        print(len(sequence_batches))
        return sequence_batches


def build_data(**config):
    """"""
    data_config = config["data"]
    data_loader = Loader(data_config, config["meta"]["run_ver"], config["meta"]["test_run"])
    # Tokenize the data and split it into batches.
    train_loader, validation_loader = data_loader.run()
    return (train_loader, validation_loader), data_loader.tokenizer.bpe_path
