import os
import glob
import json
import logging

from tqdm import tqdm


logger = logging.getLogger(__name__)


def load_videos_from_json(rootpath, channel_id=None, playlist_id=None):
    inpath = os.path.join(rootpath, 'videos_in_channel.json')
    with open(inpath, 'rt', encoding='utf-8') as infile:
        dat = json.load(infile)

    vals = []
    for data in tqdm(dat):
        assert data['kind'] == 'youtube#playlistItem'
        assert data['snippet']['resourceId']['kind'] == 'youtube#video'

        if channel_id is not None:
            assert data['snippet']['channelId'] == channel_id
            assert data['snippet']['videoOwnerChannelId'] == channel_id

        if playlist_id is not None:
            assert data['snippet']['playlistId'] == playlist_id

        vals += [{
            'id': data['snippet']['resourceId']['videoId'],
            'full': '(full documentary)' in data['snippet']['title'].lower(),
        }]

    return vals


def load_comments_from_json(rootpath, full_only=False, channel_id=None,
                            playlist_id=None):
    inpath = os.path.join(rootpath, 'comments')
    files = glob.glob(os.path.join(inpath, '*.json'))

    if full_only:
        videos = load_videos_from_json(rootpath, channel_id=channel_id,
                                       playlist_id=playlist_id)
        videos = {v['id'] for v in videos if v['full']}

    vals = []
    for file in tqdm(files):
        with open(file, 'rt', encoding='utf-8') as infile:
            comments = json.load(infile)

        for comment in comments:
            assert comment['kind'] == 'youtube#commentThread'
            assert comment['snippet']['topLevelComment']['kind'] == 'youtube#comment'
            assert comment['id'] == comment['snippet']['topLevelComment']['id']

            if channel_id is not None:
                assert comment['snippet']['channelId'] == channel_id

            video_id = comment['snippet']['videoId']
            if full_only and video_id not in videos:
                continue

            vals += [{
                'id': comment['id'],
                'video_id': video_id,
                'text': comment['snippet']['topLevelComment']['snippet']['textDisplay'],
            }]

    uniques = {}
    for c in vals:
        uniques[c['id']] = c
    uniques = list(uniques.values())

    return uniques
