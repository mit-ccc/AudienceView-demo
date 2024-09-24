#!/usr/bin/env python3

import os
import pickle
import logging
from contextlib import ExitStack

import numpy as np
import pandas as pd

import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

import utils as ut


logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self, data, output_dir, outfile_name='sentiment-scores.csv',
                 cache_file_name='sentiment-cache.pkl', padding=True,
                 truncation=True, max_length=512, batch_size=128,
                 sort_length=True, autocast=True, data_parallel=True,
                 device=None):
        super().__init__()

        self.data = data
        self.output_dir = output_dir
        self.outfile_name = outfile_name
        self.cache_file_name = cache_file_name
        self.tokenizer_args = {
            'return_tensors': 'pt',
            'padding': padding,
            'truncation': truncation,
            'max_length': max_length,
        }
        self.batch_size = batch_size
        self.sort_length = sort_length
        self.autocast = autocast
        self.data_parallel = data_parallel

        try:
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError:  # NFS
            pass

        self._cache = self._load_cache()

        #
        # Model
        #

        self.model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        if self.data_parallel and torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)

        # NOTE this causes obscure device-side errors
        # self.model = torch.compile(self.model)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model = self.model.to(self.device)

    @property
    def outfile_path(self):
        return os.path.join(self.output_dir, self.outfile_name)

    @property
    def _cache_path(self):
        return os.path.join(self.output_dir, self.cache_file_name)

    def _load_cache(self):
        if not os.path.exists(self._cache_path):
            return {}

        with open(self._cache_path, 'rb') as obj:
            return pickle.load(obj)

    def _save_cache(self):
        with open(self._cache_path, 'wb') as obj:
            with ut.DelayedKeyboardInterrupt():
                pickle.dump(self._cache, obj)

    def _is_cached(self, key):
        return key in self._cache.keys()

    def _add_to_cache(self, key, obj):
        self._cache[key] = obj

    def _get_from_cache(self, key):
        return self._cache[key]

    def tokenize(self, batch):
        return self.tokenizer(batch, **self.tokenizer_args)

    def encode(self, texts):
        inputs = self.tokenize(texts).to(self.device)

        output = self.model(**inputs, output_hidden_states=False)

        return output.logits.cpu().numpy()

    def encode_and_cache(self, batch):
        ids = batch['id']
        logits = self.encode(batch['text'])

        for i, idval in enumerate(ids):
            self._add_to_cache(idval, logits[i, :])
        self._save_cache()

        return logits

    def loader(self, data):
        if self.sort_length:
            data = sorted(data, key=lambda s: len(s['text']), reverse=True)

        return DataLoader(
            dataset=ut.SentenceDataset(data),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=10 if torch.cuda.is_available else 0,
            pin_memory=torch.cuda.is_available(),
        )

    def process(self):
        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            if self.autocast:
                device = 'cpu' if self.device == 'cpu' else 'cuda'
                stack.enter_context(torch.autocast(device))

            new = [d for d in self.data if not self._is_cached(d['id'])]
            ldr = self.loader(new)
            n_batches = int(np.ceil(len(new) / self.batch_size))

            for batch in tqdm(ldr, total=n_batches):
                self.encode_and_cache(batch)

        self._save_cache()

        ids = [d['id'] for d in self.data]
        sentiment = np.vstack([self._get_from_cache(i) for i in ids])
        sentiment = pd.DataFrame(sentiment, columns=['negative', 'neutral', 'positive'])
        sentiment['id'] = pd.Series(ids)

        sentiment.to_csv(self.outfile_path, index=False)


if __name__ == '__main__':
    from load import load_comments_from_json

    ut.log_setup()

    seed = int(os.environ.get('SEED', '42'))
    ut.seed_everything(seed)

    data_dir = os.getenv('DATA_DIR', 'data')
    output_dir = os.path.join(data_dir, 'comments')

    try:
        os.makedirs(data_dir, exist_ok=True)
    except PermissionError:  # NFS
        pass

    data = load_comments_from_json(
        rootpath=data_dir,
        channel_id=os.getenv('CHANNEL_ID', None),
        playlist_id=os.getenv('PLAYLIST_ID', None),
    )

    for d in data:
        d.pop('video_id')

    SentimentAnalyzer(data=data, output_dir=output_dir).process()
