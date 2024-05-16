import os
import json

class FileCache:
    def __init__(self, cache_dir=None, **kwargs):
        super().__init__(**kwargs)

        _cache_keys = []

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            _cache_keys += os.listdir(cache_dir)

        self.cache_dir = cache_dir
        self._cache_keys = _cache_keys

    @property
    def caching_enabled(self):
        return self.cache_dir is not None

    def is_cached(self, key):
        if not self.caching_enabled:
            return False

        return key in self._cache_keys

    def save_to_cache(self, key, obj):
        assert self.caching_enabled
        cache_path = os.path.join(self.cache_dir, key)

        with open(cache_path, 'wt', encoding='utf-8') as outfile:
            json.dump(obj, outfile)

        self._cache_keys += [key]

    def load_from_cache(self, key):
        assert self.caching_enabled
        cache_path = os.path.join(self.cache_dir, key)

        with open(cache_path, 'rt', encoding='utf-8') as infile:
            return json.load(infile)
