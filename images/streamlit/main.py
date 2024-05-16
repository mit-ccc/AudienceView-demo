#!/usr/bin/env python3

import os
import logging

import utils as ut
from app import run


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    ut.log_setup()
    ut.seed_everything()

    run()
