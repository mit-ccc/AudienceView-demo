#!/usr/bin/env python3

import os
import glob
import logging
import hashlib
from contextlib import ExitStack

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import utils as ut


logger = logging.getLogger(__name__)


class SentenceEmbedder:
    def __init__(self, data, output_dir, cache_dir='sentence-embeds-cache',
                 padding=True, truncation=True, max_length=512, batch_size=128,
                 sort_length=True, autocast=True, torch_compile=None,
                 device_ids=None):
        super().__init__()

        if torch_compile is None:
            torch_compile = bool(int(os.getenv('TORCH_COMPILE', '0')))

        self.data = data
        self.output_dir = output_dir
        self._cache_dir = cache_dir
        self.tokenizer_args = {
            'return_tensors': 'pt',
            'padding': padding,
            'truncation': truncation,
            'max_length': max_length,
        }
        self.batch_size = batch_size
        self.sort_length = sort_length
        self.autocast = autocast

        try:
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError:  # NFS
            pass

        try:
            os.makedirs(self._cache_path, exist_ok=True)
        except PermissionError:
            pass

        self._cached = self._load_cache_state()
        self.data = [r for r in self.data if not self._is_cached(r['id'])]

        #
        # Model
        #

        if device_ids is not None:
            device_ids = list(set(device_ids))

        if device_ids is None and torch.cuda.is_available():
            self.devices = [torch.device('cuda')]
        elif device_ids is None and not torch.cuda.is_available():
            self.devices = [torch.device('cpu')]
        else:
            self.devices = [torch.device(d) for d in device_ids]

        assert (
            all(d.type == 'cpu' for d in self.devices) or
            all(d.type == 'cuda' for d in self.devices)
        )

        self.model_name = 'sentence-transformers/all-mpnet-base-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        if torch_compile:
            self.model = torch.compile(self.model)

        if len(self.devices) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.devices)

        self.model = self.model.to(self.device)

    @property
    def device(self):
        return self.devices[0]

    @property
    def _cache_path(self):
        return os.path.join(self.output_dir, self._cache_dir)

    def _load_cache_state(self):
        embedded = set()

        files = glob.glob(os.path.join(self._cache_path, '*.csv'))
        for fname in files:
            embedded.update(pd.read_csv(fname)['id'].tolist())

        return embedded

    def _combine_cached_values(self):
        bases = (
            glob.glob(os.path.join(self._cache_path, '*.csv')) +
            glob.glob(os.path.join(self._cache_path, '*.pt'))
        )
        bases = [os.path.basename(b).split('.')[0] for b in bases]
        bases = list(set(bases))

        ids, embeds = [], []
        for base in tqdm(bases):
            fname = os.path.join(self._cache_path, base)

            ids += [pd.read_csv(fname + '.csv')]

            with open(fname + '.pt', 'rb') as obj:
                embeds += [torch.load(obj, 'cpu', weights_only=True)]
        ids = pd.concat(ids, axis=0)
        embeds = torch.cat(embeds, dim=0)

        return ids, embeds

    def _is_cached(self, key):
        return key in self._cached

    def _save_to_cache(self, ids, embeds):
        base = hashlib.sha1(str(ids).encode('utf-8')).hexdigest()
        ids_path = os.path.join(self._cache_path, base + '.csv')
        embeds_path = os.path.join(self._cache_path, base + '.pt')

        with open(ids_path, 'wt', encoding='utf-8') as obj:
            with ut.DelayedKeyboardInterrupt():
                pd.Series(ids, name='id').to_csv(obj)

        with open(embeds_path, 'wb') as obj:
            with ut.DelayedKeyboardInterrupt():
                torch.save(embeds.cpu(), obj)

    def tokenize(self, batch):
        return self.tokenizer(batch, **self.tokenizer_args)

    def encode(self, texts):
        inputs = self.tokenize(texts).to(self.device)
        output = self.model(**inputs).pooler_output
        output = F.normalize(output, p=2, dim=1)

        return output

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
                stack.enter_context(torch.autocast(self.device.type))

            new = [d for d in self.data if not self._is_cached(d['id'])]
            ldr = self.loader(new)
            n_batches = int(np.ceil(len(new) / self.batch_size))

            for batch in tqdm(ldr, total=n_batches):
                embeds = self.encode(batch['text'])
                self._save_to_cache(batch['id'], embeds)

        ids, embeds = self._combine_cached_values()

        ids_path = os.path.join(self.output_dir, 'sentence-embeds-ids.csv')
        embeds_path = os.path.join(self.output_dir, 'sentence-embeds.pt')

        with open(ids_path, 'wt', encoding='utf-8') as obj:
            with ut.DelayedKeyboardInterrupt():
                ids.to_csv(obj, index=False)

        with open(embeds_path, 'wb') as obj:
            with ut.DelayedKeyboardInterrupt():
                torch.save(embeds, obj)


if __name__ == '__main__':
    from load import load_comments_from_json

    ut.log_setup()

    seed = int(os.environ.get('SEED', '42'))
    ut.seed_everything(seed)

    data_dir = os.getenv('DATA_DIR', 'data')
    output_dir = os.path.join(data_dir, 'comment-topics')

    try:
        os.makedirs(data_dir, exist_ok=True)
    except PermissionError:  # NFS
        pass

    data = load_comments_from_json(
        rootpath=data_dir,
        channel_id=os.getenv('CHANNEL_ID', None),
        playlist_id=os.getenv('PLAYLIST_ID', None),
        full_only=True,
    )

    for d in data:
        d.pop('video_id')

    SentenceEmbedder(data=data, output_dir=output_dir).process()
