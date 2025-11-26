from enum import Enum


class TrackState(Enum):
    Tracked = 0
    Lost = 1
    Removed = 2


class BaseTrack:
    _count = 0

    @classmethod
    def next_id(cls):
        cls._count += 1
        return cls._count
