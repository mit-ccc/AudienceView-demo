import os
import logging
from datetime import datetime, timezone as tz


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


def days_ago(dt):
    now = datetime.now(tz.utc)
    then = datetime.strptime(dt, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=tz.utc)

    return (now - then).days
