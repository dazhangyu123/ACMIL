import os.path
from .OtherSlide import OtherSlide
from .KfbSlide import KfbSlide

from .LRUCacheDict import LRUCacheDict
from threading import Lock
slides = LRUCacheDict()
_dict_lock = Lock()


# 公共方法，打开一个切片，直接返回合适的对象
def openSlide(filename):
    ext = os.path.splitext(filename)[1][1:].lower()

    if filename in slides:
        return slides[filename]

    with _dict_lock:
        if filename in slides:
            return slides[filename]

        # print("新加载一张切片：" + filename)

        slide = None
        if ext == 'kfb':#宁波江丰      
            slide = KfbSlide(filename)
        else:# open slide
            slide = OtherSlide(filename)

        slides[filename] = slide
        # print("切片加载完成：" + filename)
        return slide

def clearCache():
    slides.clear()