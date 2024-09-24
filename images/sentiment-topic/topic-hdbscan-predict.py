#!/usr/bin/env python3

import os
import pickle
import logging

import numpy as np

from hdbscan.prediction import approximate_predict

import utils as ut


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    ut.log_setup()

    seed = int(os.environ.get('SEED', '42'))
    ut.seed_everything(seed)

    data_dir = os.getenv('DATA_DIR', 'data')

    path_u50d = os.path.join(data_dir, 'comment-topics/umap-embeds-50d.npy')
    with open(path_u50d, 'rb') as f:
        umap_embeds_50d = np.load(f)

    path_c50d = os.path.join(data_dir, 'comment-topics/hdbscan-clusterer-umap-50d.pkl')
    with open(path_c50d, 'rb') as f:
        clusterer = pickle.load(f)

    labels = approximate_predict(clusterer, umap_embeds_50d)[0]

    with ut.DelayedKeyboardInterrupt():
        path_l50d = os.path.join(data_dir, 'comment-topics/hdbscan-labels-umap-50d.npy')
        with open(path_l50d, 'wb') as f:
            np.save(f, labels)
