from collections import namedtuple, OrderedDict
import time
class LRUCacheDict(object):
    def __init__(self, expiration=15*60, maxsize=12):
        self.expiration = expiration
        self.maxsize = maxsize
        self.__expire_times = OrderedDict()
        self.__values = {}

    def __contains__(self, item):
        if self.__values.get(item):
            return True
        else:
            return False

    def __setitem__(self, key, value):
        t = int(time.time())
        self.__delitem__(key)
        self.__values[key] = value
        self.__expire_times[key] = t + self.expiration
        self.cleanup()

    def __getitem__(self, key):
        self.__expire_times[key] = int(time.time()) + self.expiration
        self.cleanup()
        return self.__values.get(key)

    def __delitem__(self, key):
        if key in self.__values:
            del self.__values[key]
            del self.__expire_times[key]

    def size(self):
        return len(self.__values)

    def clear(self):
        self.__values.clear()
        self.__expire_times.clear()

    def cleanup(self):
        t = int(time.time())
        expire_times_copy = self.__expire_times.copy()
        for key, expire in expire_times_copy.items():
            if expire < t:
                self.__delitem__(key)

        while self.size() > self.maxsize:
            for key in self.__expire_times:
                self.__delitem__(key)
                break