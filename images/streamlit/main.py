#!/usr/bin/env python3

import os
import logging

import utils as ut
from app import run

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    try:
        os.makedirs(ut.get_data_paths[0], exist_ok=True)
    except PermissionError:  # NFS
        pass

    ut.log_setup()
    ut.seed_everything()

    run()
