"""

Utilities for the custom kernels, including pre-sorting the indices.

"""
import torch
import threading
from functools import partial
import collections
from typing import List


@torch.jit.script
def get_id_and_starts(key):
    n_items = key.shape[0]
    key_diff = key[:-1] - key[1:]
    key_start = torch.nonzero(key_diff)[:, 0] + 1
    start = torch.zeros((1,), device=key.device, dtype=torch.long)
    end = n_items * torch.ones((1,), device=key.device, dtype=torch.long)
    key_start = torch.cat([start, key_start, end])
    key_ids = key[key_start[:-1]]
    return key_ids, key_start


@torch.jit.script
def resort_pairs(key, others: List[torch.Tensor]):
    keysort, argsort = torch.sort(key)
    others = [o[argsort] for o in others]
    key_ids, key_starts = get_id_and_starts(keysort)
    return argsort, key_ids, key_starts, keysort, others


class _CacheEntry:
    def __init__(self, key, cache):
        # Cache key is stored this way because sometimes
        # pytorch 'resurrects' a Tensor from C++ to pytorch
        # during an autograd calculation as a completely
        # new pyobject, so id(key) is not a completely safe way
        # to know if it's the same tensor.
        # This effect seems to go away when tensor is cached
        # (possibly because the pyobject is not garbage collected)
        # But can we do better to ensure we have the exact same data
        # as in the previous cache?
        # This could pose a problem if, for example,
        # we wrote new pairs in-place during MD and there
        # happened to be the same number with a different structure.
        # Explicitly checking that the key is exactly equal
        # is problematic because it requires device synchronization
        # in order to proceed in the interpreter.
        self.cache_key = key.data_ptr(), key.shape[0], key.device
        self.key = key
        self.cache = cache
        self.computed = False

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.key is other.key:
            return True
        if self.cache_key == other.cache_key:
            # For now raise an error since this feature is experimental.
            # Most of the benefits seem to work from the `is` check.
            if not torch.equal(self.key, other.key):
                raise RuntimeError("Caching by key does not work! Please raise an issue.")
            return True
        return False

    def find(self):
        if self in self.cache:
            other = self.cache[self.cache.index(self)]
            return other
        for other in self.cache:
            # Expensive, but not as expensive as sorting + nonzero,
            # which is also a blocking operation.
            # Note that pytorch is smart enough not to generate
            # a blocking call unless the actual memory needs to be
            # checked, i.e. it will skip if arrays have different sizes.
            if torch.equal(self.key, other.key):
                return other
        else:
            return self

    def compute_and_store(self):
        if self.computed:
            # This shouldn't happen! The guard is here because
            # this code path could activate if someone calls this class in the
            # wrong way from outside the clss.
            raise RuntimeError("A key that was marked computed was asked to be re-computed.")
        keysort, argsort = torch.sort(self.key)
        key_ids, key_starts = get_id_and_starts(keysort)
        self.argsort = argsort
        self.key_ids = key_ids
        self.key_starts = key_starts
        self.computed = True
        self.cache.append(self)

    def retrieve(self, others):
        keysort = self.key[self.argsort]
        others = [o[self.argsort] for o in others]
        return self.argsort, self.key_ids, self.key_starts, keysort, others

    @classmethod
    def lookup_key(cls, key, cache):
        entry = cls(key, cache)
        return entry.find()


N_CACHED_KEYS_PER_DEVICE = 2


def _make_cache():
    deque = collections.deque(maxlen=N_CACHED_KEYS_PER_DEVICE)
    lock = threading.Lock()
    return deque, lock


# Dict mapping device to key cache info
_CACHE_STORE = collections.defaultdict(_make_cache)


def clear_pair_cache():
    _CACHE_STORE.clear()


CACHE_LOCK_MISSES = 0


def resort_pairs_cached(key, others):
    global CACHE_LOCK_MISSES
    deque, lock = _CACHE_STORE[key.device]
    got_lock = lock.acquire(blocking=False)
    if got_lock:
        entry = _CacheEntry.lookup_key(key, deque)
        if not entry.computed:
            entry.compute_and_store()
        lock.release()
        return entry.retrieve(others)
    CACHE_LOCK_MISSES += 1
    return resort_pairs(key, others)
