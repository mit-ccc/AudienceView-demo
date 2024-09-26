# pylint: disable=too-many-instance-attributes

import os
import csv
import glob
import json
import logging
import datetime
import statistics

from tqdm import tqdm

import numpy as np
from scipy.special import softmax

from sqlalchemy import func, select, join, literal, cast, create_engine, event
from sqlalchemy import Column, ForeignKey
from sqlalchemy import Integer, String, Boolean, DateTime, Float
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, aliased

import utils as ut

logger = logging.getLogger(__name__)


Parent = declarative_base()


class Base(Parent):
    __abstract__ = True

    @classmethod
    def bulk_insert(cls, session, data):
        session.bulk_insert_mappings(cls, data)

    @classmethod
    def insert_or_update(cls, session, data):
        for val in tqdm(data):
            cobj = session.query(cls).filter_by(id=val['id']).first()
            if not cobj:
                cobj = cls()

            for k, v in val.items():
                setattr(cobj, k, v)

            session.add(cobj)


class Video(Base):
    __tablename__ = 'video'

    id = Column(String, primary_key=True)
    playlist_id = Column(String, nullable=False)
    owner_id = Column(String, nullable=False)
    kind = Column(String, nullable=False)
    publish_dt = Column(DateTime, nullable=False)

    title = Column(String, nullable=False)
    short_title = Column(String, nullable=False)

    description = Column(String, nullable=False)
    short_description = Column(String, nullable=False)

    thumbnail = Column(String, nullable=False)

    view_cnt = Column(Integer, nullable=False)
    favorite_cnt = Column(Integer, nullable=True)
    like_cnt = Column(Integer, nullable=True)
    api_comment_cnt = Column(Integer, nullable=True)

    stats = relationship('VideoStats', back_populates='video')
    comments = relationship('Comment', back_populates='video')

    @hybrid_property
    def is_full_documentary(self):
        return '(full documentary)' in self.title.lower()

    @is_full_documentary.expression
    def is_full_documentary(cls):
        return func.lower(cls.title).like('%(full documentary)%')

    @hybrid_property
    def num_comments(self):  # python-side
        return len(self.comments)

    @num_comments.expression
    def num_comments(cls):  # sql-side
        return (select([func.count(Comment.id)])
                .where(Comment.video_id == cls.id)
                .label('num_comments'))

    @staticmethod
    def validate_json(data):
        assert data['kind'] == 'youtube#playlistItem'
        assert data['snippet']['resourceId']['kind'] == 'youtube#video'

        # assert data['snippet']['channelId'] == channel_id
        # assert data['snippet']['videoOwnerChannelId'] == channel_id
        # assert data['snippet']['playlistId'] == playlist_id

    @staticmethod
    def get_best_thumbnail(data):
        thumbnails = {
            (int(v['width']), int(v['height'])): v['url']
            for k, v in data['snippet']['thumbnails'].items()
        }

        selected = sorted(
            thumbnails.keys(),
            key=lambda x: x[0] * x[1],
            reverse=True
        )[0]

        return thumbnails[selected]

    @classmethod
    def load_from_json(cls, session, path):
        with open(path, 'rt', encoding='utf-8') as infile:
            dat = json.load(infile)

        vals = []
        for data in tqdm(dat):
            cls.validate_json(data)

            vals += [{
                'id': data['snippet']['resourceId']['videoId'],
                'playlist_id': data['snippet']['playlistId'],
                'owner_id': data['snippet']['videoOwnerChannelId'],
                'kind': data['kind'],
                'publish_dt': datetime.datetime.strptime(data['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),

                'title': data['snippet']['title'],
                'short_title': data['snippet']['title']
                    .split(' | ')[0]
                    .replace(' (full documentary)', ''),

                'description': data['snippet']['description'],
                'short_description': ut.abbreviate(data['snippet']['description']),

                'thumbnail': cls.get_best_thumbnail(data),

                'view_cnt': data['statistics']['viewCount'],
                'like_cnt': data['statistics'].get('likeCount', None),
                'favorite_cnt': data['statistics'].get('favoriteCount', None),
                'api_comment_cnt': data['statistics'].get('commentCount', None),
            }]

        cls.bulk_insert(session, vals)


# used for both Comment and Commenter
def validate_comment(data):
    assert data['kind'] == 'youtube#commentThread'
    assert data['snippet']['topLevelComment']['kind'] == 'youtube#comment'
    assert data['id'] == data['snippet']['topLevelComment']['id']
    # assert data['snippet']['channelId'] == channel_id


class Commenter(Base):
    __tablename__ = 'commenter'

    id = Column(String, primary_key=True, comment='channel ID')
    channel_url = Column(String, nullable=False)
    display_name = Column(String, nullable=False)
    profile_image_url = Column(String, nullable=False)

    comments = relationship('Comment', back_populates='commenter')
    stats = relationship('CommenterStats', back_populates='commenter')

    @classmethod
    def load_from_json(cls, session, rootpath):
        path = os.path.join(rootpath, 'comments')
        files = glob.glob(os.path.join(path, '*.json'))

        commenters = set()
        for file in tqdm(files):
            with open(file, 'rt') as infile:
                comments = json.load(infile)

            for comment in comments:
                validate_comment(comment)

                author_info = comment['snippet']['topLevelComment']['snippet']

                # some kind of anon commenter has ''
                if author_info['authorChannelUrl'] != '':
                    commenters.add((
                        ('channel_url', author_info['authorChannelUrl']),
                        ('display_name', author_info['authorDisplayName']),
                        ('id', author_info['authorChannelId']['value']),
                        ('profile_image_url', author_info['authorProfileImageUrl']),
                    ))

        commenters = [dict(c) for c in commenters]
        unique_commenters = {}
        for c in commenters:
            unique_commenters[c['id']] = c
        unique_commenters = unique_commenters.values()

        cls.bulk_insert(session, unique_commenters)


class Cluster(Base):
    __tablename__ = 'cluster'

    id = Column(String, primary_key=True)
    load_dt = Column(DateTime, nullable=False, default=func.now())

    comments = relationship('Comment', back_populates='cluster')

    @classmethod
    def load_from_json(cls, session, rootpath):
        clusters_path = os.path.join(rootpath, 'comment-topics', 'hdbscan-labels-umap-50d.npy')
        with open(clusters_path, 'rb') as f:
            clusters = np.load(f)

        clusters = list(set(clusters.tolist()))
        clusters = [{'id': 'cluster:' + str(c)} for c in clusters]

        cls.bulk_insert(session, clusters)


class Comment(Base):
    __tablename__ = 'comment'

    id = Column(String, primary_key=True)
    text = Column(String, nullable=False)
    publish_dt = Column(DateTime, nullable=False)
    update_dt = Column(DateTime, nullable=False)
    like_cnt = Column(Integer, nullable=False)
    viewer_rating = Column(String, nullable=False)

    can_reply = Column(Boolean, nullable=False)
    total_reply_cnt = Column(Integer, nullable=False)
    is_public = Column(Boolean, nullable=False)

    sentiment_negative = Column(Float, nullable=False)
    sentiment_neutral = Column(Float, nullable=False)
    sentiment_positive = Column(Float, nullable=False)

    cluster_id = Column(String, ForeignKey('cluster.id'), index=True,
                        nullable=True)

    commenter_id = Column(String, ForeignKey('commenter.id'),
                          index=True, nullable=True)

    video_id = Column(String, ForeignKey('video.id'),
                      index=True, nullable=False)

    commenter = relationship('Commenter', back_populates='comments')
    video = relationship('Video', back_populates='comments')
    cluster = relationship('Cluster', back_populates='comments')

    @property
    def short_text(self):  # python-side
        return ut.abbreviate(self.text)

    @hybrid_property
    def is_update(self):
        if not self.text:
            return False

        text = self.text.lower().replace('\n', ' ')

        return (
            'update' in text or
            'latest' in text or
            'follow-up' in text or
            'follow up' in text
        )

    @is_update.expression
    def is_update(cls):
        text = func.replace(func.lower(cls.text), '\n', ' ')

        return (
            text.like('%update%') |
            text.like('%latest%') |
            text.like('%follow-up%') |
            text.like('%follow up%')
        )

    @classmethod
    def load_from_json(cls, session, rootpath):
        path = os.path.join(rootpath, 'comments')

        sentiment_path = os.path.join(path, 'sentiment-scores.csv')
        with open(sentiment_path, 'rt') as infile:
            reader = csv.DictReader(infile)

            sentiment = {
                row['id']: {k: v for k, v in row.items() if k != 'id'}
                for row in reader
            }

        sample_ids_path = os.path.join(rootpath, 'comment-topics','umap-hdbscan-sample-ids.csv')
        with open(sample_ids_path, 'rt') as infile:
            reader = csv.DictReader(infile)
            sample_ids = [row['id'] for row in reader]

        clusters_path = os.path.join(rootpath, 'comment-topics', 'hdbscan-labels-umap-50d.npy')
        with open(clusters_path, 'rb') as f:
            clusters = np.load(f).tolist()
        clusters = dict(zip(sample_ids, clusters))

        vals = []
        files = glob.glob(os.path.join(path, '*.json'))
        for file in tqdm(files):
            with open(file, 'rt', encoding='utf-8') as infile:
                comments = json.load(infile)

            for comment in comments:
                validate_comment(comment)

                snippet = comment['snippet']['topLevelComment']['snippet']

                sentiments = softmax(np.asarray([
                    float(sentiment[comment['id']]['negative']),
                    float(sentiment[comment['id']]['neutral']),
                    float(sentiment[comment['id']]['positive']),
                ]))

                cluster_id = clusters.get(comment['id'], None)

                val = {
                    'id': comment['id'],
                    'video_id': comment['snippet']['videoId'],

                    'can_reply': comment['snippet']['canReply'],
                    'total_reply_cnt': comment['snippet']['totalReplyCount'],
                    'is_public': comment['snippet']['isPublic'],

                    'text': snippet['textDisplay'],
                    'like_cnt': snippet['likeCount'],
                    'viewer_rating': snippet['viewerRating'],

                    'sentiment_negative': sentiments[0],
                    'sentiment_neutral': sentiments[1],
                    'sentiment_positive': sentiments[2],

                    'cluster_id': 'cluster:' + str(cluster_id) if cluster_id is not None else None,

                    'publish_dt': datetime.datetime.strptime(snippet['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                    'update_dt': datetime.datetime.strptime(snippet['updatedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                }

                if 'authorChannelId' in snippet.keys():
                    val['commenter_id'] = snippet['authorChannelId']['value']

                vals += [val]

        uniques = {}
        for c in vals:
            uniques[c['id']] = c
        uniques = uniques.values()

        cls.bulk_insert(session, uniques)


class VideoStats(Base):
    __tablename__ = 'video_stats'

    video_id = Column(String, ForeignKey('video.id'), primary_key=True)
    is_full_documentary = Column(Boolean, primary_key=True)

    comment_cnt = Column(Integer, nullable=False)
    update_cnt = Column(Integer, nullable=False)
    view_cnt = Column(Integer, nullable=False)
    last_comment_dt = Column(DateTime, nullable=True)

    avg_sentiment_negative = Column(Float, nullable=True)
    avg_sentiment_neutral = Column(Float, nullable=True)
    avg_sentiment_positive = Column(Float, nullable=True)

    video = relationship('Video', back_populates='stats')

    @staticmethod
    def base_select_statement(full_doc, filters={}):
        stmt = select(
            Video.id.label('video_id'),
            literal(full_doc).label('is_full_documentary'),

            func.count(Comment.id).label('comment_cnt'),
            func.sum(cast(func.coalesce(Comment.is_update, False), Integer)).label('update_cnt'),
            func.max(Comment.publish_dt).label('last_comment_dt'),

            func.avg(Comment.sentiment_negative).label('avg_sentiment_negative'),
            func.avg(Comment.sentiment_neutral).label('avg_sentiment_neutral'),
            func.avg(Comment.sentiment_positive).label('avg_sentiment_positive'),
        ) \
            .select_from(Video) \
            .outerjoin(Comment, Video.id == Comment.video_id) \
            .group_by(Video.id)

        flt = []
        for key, value in filters.items():
            flt.append(getattr(Video, key) == value)
        stmt = stmt.filter(*flt)

        view_stmt = select(
            Video.id.label('video_id'),
            func.sum(Video.view_cnt).label('view_cnt')
        ) \
            .select_from(Video) \
            .group_by('video_id')

        stmt = aliased(stmt.subquery(), name='stmt')
        view_stmt = aliased(view_stmt.subquery(), name='view_stmt')

        stmt_cols = [col for col in stmt.columns if col.key != 'video_id']
        view_stmt_cols = [col for col in view_stmt.columns]
        all_cols = stmt_cols + view_stmt_cols

        join_stmt = join(stmt, view_stmt, stmt.c.video_id == view_stmt.c.video_id)

        return select(*all_cols).select_from(join_stmt)

    @classmethod
    def load_from_db(cls, session):
        stmt = cls.base_select_statement(full_doc=False)
        stats = session.execute(stmt).all()
        stats = [s._asdict() for s in stats]
        cls.bulk_insert(session, stats)

        stmt = cls.base_select_statement(
            full_doc=True,
            filters={'is_full_documentary': True}
        )
        stats = session.execute(stmt).all()
        stats = [s._asdict() for s in stats]
        cls.bulk_insert(session, stats)


class CommenterStats(Base):
    __tablename__ = 'commenter_stats'

    commenter_id = Column(String, ForeignKey('commenter.id'), primary_key=True)
    is_full_documentary = Column(Boolean, primary_key=True)

    display_name = Column(String, nullable=False)

    comment_cnt = Column(Integer, nullable=False)
    update_cnt = Column(Integer, nullable=False)
    last_comment_dt = Column(DateTime, nullable=False)

    avg_sentiment_negative = Column(Float, nullable=False)
    avg_sentiment_neutral = Column(Float, nullable=False)
    avg_sentiment_positive = Column(Float, nullable=False)

    commenter = relationship('Commenter', back_populates='stats')

    @staticmethod
    def base_select_statement(full_doc, filters={}):
        stmt = select(
            Commenter.id.label('commenter_id'),
            literal(full_doc).label('is_full_documentary'),

            func.max(Commenter.display_name).label('display_name'),

            func.count(Comment.id).label('comment_cnt'),
            func.sum(cast(Comment.is_update, Integer)).label('update_cnt'),
            func.max(Comment.publish_dt).label('last_comment_dt'),

            func.avg(Comment.sentiment_negative).label('avg_sentiment_negative'),
            func.avg(Comment.sentiment_neutral).label('avg_sentiment_neutral'),
            func.avg(Comment.sentiment_positive).label('avg_sentiment_positive'),
        ) \
            .select_from(Commenter) \
            .join(Comment, Comment.commenter_id == Commenter.id) \
            .join(Video, Video.id == Comment.video_id) \
            .group_by(Commenter.id)

        flt = []
        for key, value in filters.items():
            flt.append(getattr(Video, key) == value)
        stmt = stmt.filter(*flt)

        return stmt

    @classmethod
    def load_from_db(cls, session):
        stmt = cls.base_select_statement(full_doc=False)
        stats = session.execute(stmt).all()
        stats = [s._asdict() for s in stats]
        cls.bulk_insert(session, stats)

        stmt = cls.base_select_statement(
            full_doc=True,
            filters={'is_full_documentary': True}
        )
        stats = session.execute(stmt).all()
        stats = [s._asdict() for s in stats]
        cls.bulk_insert(session, stats)


class StdevAgg:
    def __init__(self):
        self.values = []

    def step(self, value):
        if value is not None:
            self.values.append(value)

    def finalize(self):
        if len(self.values) < 2:
            return None
        return statistics.stdev(self.values)


def populate_db(engine):
    Base.metadata.create_all(engine)

    db_factory = sessionmaker(bind=engine)

    data_dir, db_path, cache_dir = ut.get_data_paths()
    videos_path = os.path.join(data_dir, 'videos_in_channel.json')

    with db_factory() as session:
        session.begin()

        Video.load_from_json(session, videos_path)
        Commenter.load_from_json(session, data_dir)
        Cluster.load_from_json(session, data_dir)
        Comment.load_from_json(session, data_dir)
        CommenterStats.load_from_db(session)
        VideoStats.load_from_db(session)

        session.commit()


def get_db(db_path):
    db_url = 'sqlite:///' + os.path.abspath(db_path)
    engine = create_engine(db_url, echo=False)

    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("pragma foreign_keys=on")
        cursor.execute("pragma defer_foreign_keys=on")
        cursor.close()
    event.listen(engine, 'connect', set_sqlite_pragma)

    if not os.path.exists(db_path):
        populate_db(engine)

    engine.raw_connection().create_aggregate("STDEV", 1, StdevAgg)

    return sessionmaker(bind=engine)
