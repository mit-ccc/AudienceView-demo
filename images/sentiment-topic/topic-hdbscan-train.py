#!/usr/bin/env python3

import os
import pickle
import logging

import numpy as np

from hdbscan import HDBSCAN

import utils as ut


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    ut.log_setup()

    seed = int(os.environ.get('SEED', '42'))
    ut.seed_everything(seed)

    umap_embeds_50d_path = 'data/comment-topics/umap-embeds-50d.npy'
    with open(umap_embeds_50d_path, 'rb') as f:
        umap_embeds_50d = np.load(f)

    logger.info('Fitting HDBSCAN')

    params = {
        'min_cluster_size': 768,
        'min_samples': 768,
        'cluster_selection_method': 'leaf',
        'prediction_data': True,
        'core_dist_n_jobs': -1,
    }

    clusterer = HDBSCAN(**params)
    clusterer.fit(umap_embeds_50d)

    with ut.DelayedKeyboardInterrupt():
        with open('data/comment-topics/hdbscan-clusterer-umap-50d.pkl', 'wb') as f:
            pickle.dump(clusterer, f)
