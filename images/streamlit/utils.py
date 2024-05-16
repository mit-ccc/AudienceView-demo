import os
import re
import string
import random
import logging

from itertools import islice

import numpy as np

from sqlalchemy import inspect


logger = logging.getLogger(__name__)


def log_setup(lvl=None):
    try:
        if lvl is not None:
            log_level = lvl
        else:
            log_level = getattr(logging, os.environ['LOG_LEVEL'])
    except KeyError:
        log_level = logging.INFO
    except AttributeError as exc:
        raise AttributeError('Bad log level') from exc

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def seed_everything():
    seed = int(os.environ.get('SEED', '42'))

    random.seed(seed)
    np.random.seed(seed)


def remove_punctuation(text):
    for punct in string.punctuation:
        if punct == "'":
            continue

        text = text.replace(punct, '')

    return text


def abbreviate(txt):
    newtxt = ' '.join(txt.split()[0:25])
    if len(newtxt) != len(txt):  # i.e., we abbreviated it
        newtxt += '...'

    return newtxt


def remove_last_n_words(s, n):
    parts = re.findall(r'\S+\s*|\s+', s)  # each part incl trailing whitespace

    if len(parts) <= n:
        raise ValueError("Not enough words to remove")

    return ''.join(parts[:-n])


def fk_from_to(model, target_model):
    mapper = inspect(model)

    for relationship in mapper.relationships:
        if relationship.mapper.class_ == target_model:
            return relationship.local_remote_pairs[0][0].name

    return None

def pk_for(model):
    primary_keys = [key.name for key in inspect(model).primary_key]

    assert len(primary_keys) == 1

    return primary_keys[0]
