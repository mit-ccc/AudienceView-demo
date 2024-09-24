#!/usr/bin/env python3

import os
import logging

import numpy as np
import pandas as pd

import torch

from umap import UMAP

import utils as ut


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    from load import load_comments_from_json

    ut.log_setup()

    seed = int(os.environ.get('SEED', '42'))
    ut.seed_everything(seed)

    data_dir = os.getenv('DATA_DIR', 'data')

    #
    # Sample to train on
    #

    sample = load_comments_from_json(
        rootpath=data_dir,
        channel_id=os.getenv('CHANNEL_ID', None),
        playlist_id=os.getenv('PLAYLIST_ID', None),
    )

    sample = pd.DataFrame(sample) \
        .drop('text', axis=1) \
        .groupby('video_id') \
        .apply(lambda x: x.sample(min(500, x.shape[0])).reset_index(drop=True)) \
        ['id'] \
        .tolist()
    sample = set(sample)

    embeds_ids_path = os.path.join(data_dir, 'comment-topics/sentence-embeds-ids.csv')
    ids = pd.read_csv(embeds_ids_path)['id']
    train_mask = np.asarray([i in sample for i in ids.to_numpy().tolist()])
    logger.info(f'Training on {train_mask.sum()} samples')

    sample_ids_path = os.path.join(data_dir, 'comment-topics/umap-hdbscan-sample-ids.csv')
    ids.loc[train_mask].to_csv(sample_ids_path, index=False)

    #
    # 50d UMAP
    #

    logger.info('50d UMAP')

    embeds_file = os.path.join(data_dir, 'comment-topics/sentence-embeds.pt')
    with open(embeds_file, 'rb') as obj:
        embeds = torch.load(obj, 'cpu', weights_only=True) \
            .float() \
            .numpy() \
            [train_mask, ...]

    params = {
        'n_neighbors': 15,
        'min_dist': 0.1,
        'init': 'random',
        'metric': 'cosine',  # same as euclidean if vectors normalized

        'unique': True,
        'low_memory': False,
        'n_jobs': -1,

        'verbose': True,
    }

    umap_model_50d = UMAP(n_components=50, **params).fit(embeds)
    umap_embeds_50d = umap_model_50d.transform(embeds)

    with ut.DelayedKeyboardInterrupt():
        umap_embeds_50d_path = os.path.join(data_dir, 'comment-topics/umap-embeds-50d.npy')
        with open(umap_embeds_50d_path, 'wb') as f:
            np.save(f, umap_embeds_50d)
