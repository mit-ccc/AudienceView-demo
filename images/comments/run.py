#!/usr/bin/env python3

"""
Given a YouTube channel ID, this script pulls info on the videos that have been
posted to the channel and then pulls the raw comments for each video.  This
script requires that you have a YouTube Data API key (see
https://developers.google.com/youtube/v3/getting-started) and that it be set in
the YOUTUBE_API_KEY environment variable.
"""

import os
import json
import logging
import argparse

from googleapiclient.errors import HttpError
from googleapiclient.discovery import build

from tqdm import tqdm

import utils as ut


logger = logging.getLogger(__name__)


class ChannelFetch:
    def __init__(self, channel_id, outdir='data', cached_videos=False,
                 progress=True, new_threshold_days=None):
        super().__init__()

        self.channel_id = channel_id
        self.outdir = outdir
        self.cached_videos = cached_videos
        self.progress = progress
        self.new_threshold_days = new_threshold_days

        self.youtube_api_key = os.environ['YOUTUBE_API_KEY']
        self.api = build('youtube', 'v3', developerKey=self.youtube_api_key)

        os.makedirs(self.comments_path, exist_ok=True)

    @property
    def video_path(self):
        return os.path.join(args.outdir, 'videos_in_channel.json')

    @property
    def comments_path(self):
        return os.path.join(args.outdir, 'comments')

    def fetch_videos(self):
        # Get the Uploads playlist ID from channel ID
        res = self.api.channels().list(id=self.channel_id, part="contentDetails").execute()
        playlist_id = res["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        videos_basic_info = []
        next_page_token = None

        # Fetch basic video details
        while True:
            res = (
                self.youtube.playlistItems()
                .list(
                    playlistId=playlist_id,
                    part="snippet",
                    maxResults=50,
                    pageToken=next_page_token,
                )
                .execute()
            )

            videos_basic_info += res["items"]
            next_page_token = res.get("nextPageToken")

            if next_page_token is None:
                break

        # Extract video IDs
        video_ids = [video["snippet"]["resourceId"]["videoId"] for video in videos_basic_info]

        # Fetch statistics for each video
        videos_with_stats = []
        for i in range(0, len(video_ids), 50):  # API allows max 50 ids per request
            res = self.api.videos().list(
                id=",".join(video_ids[i:i+50]),
                part="snippet,statistics"
            ).execute()
            videos_with_stats += res["items"]

        # Combine basic info and statistics
        for video in videos_basic_info:
            video_id = video["snippet"]["resourceId"]["videoId"]
            for stats_video in videos_with_stats:
                if stats_video["id"] == video_id:
                    video["statistics"] = stats_video.get("statistics", {})

        return videos_basic_info

    def fetch_video_comments(self, video_id, max_results=100):
        kwargs = {
            "part": "snippet",
            "maxResults": max_results,
            "videoId": video_id,
            "textFormat": "plainText",
            "order": "time",
        }

        try:
            results = self.api.commentThreads().list(**kwargs).execute()
        except HttpError:
            results = []

        comments = []
        while results:
            for item in results["items"]:
                comments.append(item)

            # check if there are more comments
            if "nextPageToken" in results:
                args["pageToken"] = results["nextPageToken"]
                results = self.api.commentThreads().list(**args).execute()
            else:
                break

        return comments

    def ensure_videos(self):
        if os.path.exists(self.video_path) and self.cached_videos:
            with open(self.video_path, 'rt', encoding='utf-8') as f:
                videos = json.load(f)
        else:
            videos = self.fetch_videos()
            with open(self.video_path, 'wt', encoding='utf-8') as f:
                json.dump(videos, f)

        return videos

    def ensure_comments(self, video):
        video_id = video['snippet']['resourceId']['videoId']
        age_in_days = ut.days_ago(video['snippet']['publishedAt'])
        outfile = os.path.join(self.comments_path, f'{video_id}.json')

        if (
            os.path.exists(outfile) and
            self.new_threshold_days is not None and
            age_in_days > self.new_threshold_days
        ):
            return

        try:
            comments = self.fetch_video_comments(video_id)
        except HttpError:
            logger.exception('Could not load video info: %s', video_id)
        else:
            with open(outfile, 'wt', encoding='utf-8') as f:
                json.dump(comments, f)

        return comments

    def run(self):
        videos = self.ensure_videos()

        pbar = tqdm(videos, disable=(not self.progress))
        for video in pbar:
            video_id = video['snippet']['resourceId']['videoId']
            pbar.set_description(video_id)

            self.ensure_comments(video)


def parse_args():
    parser = argparse.ArgumentParser(description='Fetch YouTube video/comment data')

    parser.add_argument('channel_id', nargs='?', default=None, help='YouTube Channel ID')

    parser.add_argument('--cached', '-c', action='store_true', help='Avoid fetching new videos (comments only)')
    parser.add_argument('--outdir', '-o', default='data', help='Output directory')
    parser.add_argument('--verbose', '-d', action='store_true', help='Debug output')
    parser.add_argument('--progress', '-p', action='store_true', help='Progress bar')
    parser.add_argument('--new-threshold-days', '-n', default=None, help='Always refresh comments for videos newer than this (in days)')

    return parser.parse_args()


if __name__ == '__main__':
    args = vars(parse_args())

    ut.log_setup('DEBUG' if args.pop('verbose') else None)

    if args['channel_id'] is None:
        assert 'CHANNEL_ID' in os.environ.keys()
        args['channel_id'] = os.environ.get('CHANNEL_ID')

    ChannelFetch(**args).run()
