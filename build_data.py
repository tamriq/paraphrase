import math
import random
import os
import zipfile

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
        pads_enc = [self.pad_index] * (self.max_len+2 - len(enc))
        pads_dec = [self.pad_index] * (max_len_dec+2 - len(dec))
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


class TokenizerBatcher:
    """
    Prepares the data for pytorch DataLoader:
        - loads zipped tsv file
        - trains the BPE tokenizer on the data
        - tokenizes the data and separates it into batches
    """
    def __init__(self, dataset_name: str, vocab_size: int):
        """

        :param dataset_name:
        :param vocab_size:
        """
        self.dataset_df = self.load_data(dataset_name)
        bpe_file_name = self.write_txt_data(self.dataset_df, dataset_name)
        self.tokenize = self.train_tokenizer(dataset_name, bpe_file_name, vocab_size)

    def run(self, batch_size: int, sequence_bucketing: bool):
        tokenized_pairs = self.tokenize_data()
        batches = self.batch_data(tokenized_pairs, batch_size, sort_by_src=sequence_bucketing)
        random.shuffle(batches)
        return batches

    def tokenize_data(self, bpe_batch_size: int = 256):
        """

        :param bpe_batch_size:
        :return:
        """
        tokenized_src = []
        tokenized_trg = []
        # Iterate over the dataframe and tokenize the pairs.
        for i_batch in tqdm(range(math.ceil(len(self.dataset_df) / bpe_batch_size))):
            # Tokenize source and target sentences of the bpe-batch separately.
            src_batch = self.tokenize.encode(list(self.dataset_df[0][i_batch * bpe_batch_size:(i_batch + 1) * bpe_batch_size]))
            trg_batch = self.tokenize.encode(list(self.dataset_df[1][i_batch * bpe_batch_size:(i_batch + 1) * bpe_batch_size]))
            # Add the sentences tokens to the storage list.
            tokenized_src.extend(src_batch)
            tokenized_trg.extend(trg_batch)
            # Store the source and target sentences tokens in reverse order to learn all possible combinations.
            tokenized_src.extend(trg_batch)
            tokenized_trg.extend(src_batch)
        # Combine back into pairs.
        tokenized_pairs = [i for i in zip(tokenized_src, tokenized_trg)]
        random.shuffle(tokenized_pairs)
        return tokenized_pairs

    @staticmethod
    def batch_data(tokenized_pairs, batch_size: int, sort_by_src: bool):
        """

        :param tokenized_pairs:
        :param batch_size:
        :param sort_by_src:
        :return:
        """
        if sort_by_src:
            # Sort the pairs by the length of the source sentence.
            # This type of sorting is required to implement the SequenceBucketing algorithm.
            tokenized_pairs = sorted(tokenized_pairs, key=lambda x: len(x[0]), reverse=True)
        batches = []
        for i_batch in range(math.ceil(len(tokenized_pairs) / batch_size)):
            batches.append(tokenized_pairs[i_batch * batch_size:(i_batch + 1) * batch_size])
        # LOG LENGTH OF BATCHES
        print(len(batches))
        return batches

    @staticmethod
    def train_tokenizer(dataset_name, bpe_file_name, vocab_size):
        """

        :param dataset_name:
        :param bpe_file_name:
        :param vocab_size:
        :return:
        """
        model_path = f"bpe_{dataset_name}.model"
        yttm.BPE.train(data=bpe_file_name, vocab_size=vocab_size, model=model_path)
        bpe_tokenizer = yttm.BPE(model=model_path)
        return bpe_tokenizer

    @staticmethod
    def load_data(name: str):
        """
        Load and unzip the data.

        :param name: the name of the dataset
            - 'backed' for the corpus of the backtranslated pairs
            - 'news' for the corpus of the paired news titles
            - 'subtitles' for the paraphrase corpus of subtitles
        :return: data in DataFrame format
        """
        name = name.strip()
        assert name not in ["backed", "news", "subtitles"], "Invalid data name"
        tsv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"data/{name}.tsv")
        # Unzip tsv file.
        loaded_zip = zipfile.ZipFile(f"{tsv_path}.zip")
        loaded_zip.extractall()
        df = pd.read_csv(tsv_path, sep='\t', header=None, error_bad_lines=False)
        # Drop Nans.
        df_no_nan = df.dropna()
        # Shuffle data.
        df_no_nan = df_no_nan.sample(frac=1).reset_index(drop=True)
        return df_no_nan

    @staticmethod
    def write_txt_data(dataset_df: pd.DataFrame, dataset_name: str) -> str:
        """

        :param dataset_df: data in dataframe format
        :param dataset_name: the name of the dataset
        :return:
        """
        err = 0
        bpe_file_name = f"for_bpe_{dataset_name}.txt"
        f = open(bpe_file_name, 'w')
        for src in dataset_df[0]:
            try:
                f.write(src + '\n')
            except:
                err += 1
        for trg in dataset_df[1]:
            try:
                f.write(trg + '\n')
            except:
                err += 1
        f.close()
        # We need to log here?
        print(err)
        return bpe_file_name


def get_data_loader_train_val(**config):
    """"""
    bpe_batcher = TokenizerBatcher(config["dataset_name"], config["vocab_size"])
    # Tokenize the data and split it into batches.
    batches = bpe_batcher.run(batch_size=config["batch_size"], sequence_bucketing=config["sequence_bucketing"])
    # Compute the start index for validation dataset.
    validation_start_index = int(len(batches) * config["train_val_proportion"])
    # Get loaders.
    train_loader = SequenceBucketingData(batches[:-validation_start_index], config["max_len"], config["pad_idx"],
                                         config["eos_idx"], config["bos_idx"])
    validation_loader = SequenceBucketingData(batches[-validation_start_index:], config["max_len"], config["pad_idx"],
                                              config["eos_idx"], config["bos_idx"])
    return train_loader, validation_loader

