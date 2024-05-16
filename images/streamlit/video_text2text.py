from typing import Optional, Dict, Any

import re
import copy
import random
import logging
import hashlib
import collections

from abc import abstractmethod
from functools import cached_property

import openai

import sqlalchemy as sa
from sqlalchemy import func

import utils as ut
from file_cache import FileCache
from models import Base, Video, Comment, Cluster


logger = logging.getLogger(__name__)


class MissingDataException(Exception):
    pass


class Text2Text(collections.abc.Mapping, FileCache):
    '''
    Text2text tasks on videos
    '''

    def __init__(self, prompt: str, session: sa.orm.session.Session,
                 target: Base, filters: Dict[str, Any] = {},
                 max_comments: Optional[int] = 100,
                 model: str = 'gpt-4-1106-preview',
                 temperature: float = 0.0,
                 max_tokens: int = 1000,
                 cache_dir: Optional[str] = None, **kwargs):
        super().__init__(cache_dir=cache_dir)

        self.prompt = prompt
        self._session = session
        self.target = target
        self.filters = filters

        self.max_comments = max_comments

        self._openai_kwargs = dict(
            kwargs,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        assert len(self._keys) == len(set(self._keys))

    def __repr__(self):
        cls = self.__class__.__name__

        return f'{cls}(prompt=\'\'\'{self.prompt}\'\'\', ' + \
               f'engine=Engine(\'{self._session.get_bind().url}\')'

    @property
    def pk(self):
        return ut.pk_for(self.target)

    @cached_property
    def _keys(self):
        pk = ut.pk_for(self.target)

        query = self._session \
            .query(getattr(self.target, pk))

        if self.filters:
            flt = []
            for key, value in self.filters.items():
                if hasattr(self.target, key):
                    flt.append(getattr(self.target, key) == value)
            query = query.filter(*flt)

        return [k[0] for k in query.all()]

    @property
    def cached_keys(self):
        return [
            vi
            for vi in self._keys
            if self.is_cached(self._get_cache_key(vi))
        ]

    @property
    def uncached_keys(self):
        return [
            vi
            for vi in self._keys
            if not self.is_cached(self._get_cache_key(vi))
        ]

    def __contains__(self, key):
        return key in self._keys

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __getitem__(self, key):
        assert key in self.keys()
        return self.text2text(key)

    def _get_cache_key(self, key=None):
        if key is None:
            key = 'None'

        val = [self.prompt, str(self._openai_kwargs)]
        if key is None and self.filters:
            val += [self.filters]
        val = tuple(val)
        val = (str(val) + '.gpt').encode('utf-8')

        kls = type(self).__name__

        return f'key={key}__kls={kls}__hash={hashlib.sha1(val).hexdigest()}'

    @abstractmethod
    def promptify(self, key=None):
        '''
        Combine the prompt and video data for LLM
        '''

        raise NotImplementedError()

    def api_query(self, prompt):
        '''
        Run a query against the OpenAI chat.completions API.
        '''

        # auto-retries w/ backoff if error
        return openai.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **self._openai_kwargs
        ).choices[0].message.content

    def text2text(self, key=None):
        '''
        Get text2text response for a video
        '''

        cache_key = self._get_cache_key(key)
        if self.is_cached(cache_key):
            return self.load_from_cache(cache_key)

        try:
            prompt = self.promptify(key)
            resp = self.api_query(prompt)

            ret = {
                'prompt': prompt,
                'response': resp
            }
        except MissingDataException:
            ret = {'prompt': '', 'response': ''}

        self.save_to_cache(cache_key, ret)

        return ret


class CommentsText2Text(Text2Text):
    '''
    Text2text based on comments
    '''

    def __init__(self, min_comments: int = 20, max_words: int = 20_000,
                 **kwargs):
        assert min_comments > 0

        super().__init__(**kwargs)

        self.min_comments = min_comments
        self.max_words = max_words

    @property
    def fk(self):
        return ut.fk_from_to(Comment, self.target)

    def _n_comments(self, key=None):
        assert key is None or key in self.keys()
        query_keys = self._keys if key is None else [key]

        return self._session \
            .query(func.count(getattr(Comment, self.pk))) \
            .filter(getattr(Comment, self.fk).in_(query_keys)) \
            .scalar()

    def _get_comments(self, key=None):
        assert key is None or key in self.keys()
        query_keys = self._keys if key is None else [key]

        query = self._session \
            .query(Comment) \
            .filter(getattr(Comment, self.fk).in_(query_keys)) \
            .order_by(func.random())

        if self.max_comments is not None:
            query = query.limit(self.max_comments)

        return [
            {'id': c.id, 'text': c.text}
            for c in query.all()
        ]

    @staticmethod
    def _format_comments(comments):
        ret = []

        for i, c in enumerate(comments):
            num = i + 1
            txt = c['text'][:200]

            ret += [f'- [{num}] "{txt}"']

        return '\n'.join(ret)

    @staticmethod
    def _comments_from_prompt(prompt):
        prompt = prompt.strip().split('\n\n')[1:]  # drop instructions
        prompt = '\n\n'.join(prompt)

        comments = re.split(r'-\s*(?=\[\d+\])', prompt)
        if comments and comments[0] == '':
            comments = comments[1:]
        comments = [c.lstrip('-').strip() for c in comments]
        comments = {c.split(' ')[0]: c for c in comments}

        return comments

    @staticmethod
    def _trim_to_max_words(comments, max_words):
        wc_total = sum(len(c['text'].split()) for c in comments)
        while wc_total > max_words:
            comments = comments[1:]
            wc_total = sum(len(c['text'].split()) for c in comments)

        return comments

    def _trim_comments(self, comments):
        if self.max_comments is not None:
            comments = comments[:self.max_comments]

        if self.max_words is not None:
            comments = self._trim_to_max_words(comments, self.max_words)

        return comments

    def formatted_comments(self, key=None):
        '''
        Get sampled, formatted comments for use in LLM prompt
        '''
        comments = self._get_comments(key)
        random.shuffle(comments)

        comments = self._trim_comments(comments)

        return self._format_comments(comments)

    def promptify(self, key=None):
        assert key is None or key in self.keys()

        if self._n_comments(key) < self.min_comments:
            raise MissingDataException('Too few comments')

        return '\n\n'.join([self.prompt, self.formatted_comments(key)])


class VideoSummary(CommentsText2Text):
    '''
    Summary of video comments
    '''

    def __init__(self, **kwargs):
        prompt = 'These are comments from a video, in no particular ' + \
                 'order. Please summarize the major themes of the  ' + \
                 'comments, and please cite examples.'

        super().__init__(prompt=prompt, target=Video, **kwargs)


class VideoSuggestions(CommentsText2Text):
    '''
    Suggestions for new Frontline content
    '''

    def __init__(self, **kwargs):
        prompt = 'These are comments from a Frontline documentary video. ' + \
                 'Based on these comments, what kinds of new documentary ' + \
                 'content should Frontline work on?  Please cite examples ' + \
                 'of comments that support your answer.'

        super().__init__(prompt=prompt, target=Video, **kwargs)


class ClusterShortName(CommentsText2Text):
    '''
    Summary of topic cluster comments
    '''

    def __init__(self, **kwargs):
        prompt = (
            'These are samples from a cluster of related comments on ' +
            'a group of videos, in no particular order. Please ' +
            'summarize the common topic of the comments in a short ' +
            'phrase. For example, you might say "immigration" or ' +
            '"support for Democratic candidates" or "effects of the Trump ' +
            'presidency". The short phrase should appear alone on the last ' +
            'line of your response. If there is no common topic (i.e., the ' +
            'comments are too varied), please put "no topic" on the last line.'
        )

        super().__init__(prompt=prompt, target=Cluster, **kwargs)
