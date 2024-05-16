#!/usr/bin/env python3

import os
import logging

import utils as ut
from models import get_db
from app import DB_PATH, SUMMARY_SUGGEST_CACHE_DIR
from video_text2text import VideoSummary, VideoSuggestions, ClusterShortName


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    ut.log_setup()
    ut.seed_everything()

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    db_factory = get_db(DB_PATH)

    with db_factory() as db_session:
        kwargs = {'session': db_session, 'cache_dir': SUMMARY_SUGGEST_CACHE_DIR}

        clusters = ClusterShortName(**kwargs)
        for key in clusters.uncached_keys:
            clusters.text2text(key)

        summaries = VideoSummary(filters={'is_full_documentary': True}, **kwargs)
        for key in summaries.uncached_keys:
            summaries.text2text(key)
        summaries.text2text(None)

        suggestions = VideoSuggestions(filters={'is_full_documentary': True}, **kwargs)
        for key in suggestions.uncached_keys:
            suggestions.text2text(key)
        suggestions.text2text(None)
