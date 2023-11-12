# from openslide import AbstractSlide, _OpenSlideMap
import pdb

from . import kfb_lowlevel
from .kfb_deepzoom import KfbDeepZoomGenerator

import operator, math
import numpy as np
from io import BytesIO
from PIL import Image
from collections import Mapping
class kfbRef:
    img_count = 0


class KfbSlide():
    def __init__(self, filename):
        # AbstractSlide.__init__(self)
        self.__filename = filename
        self._osr = kfb_lowlevel.kfbslide_open(filename)
        # pdb.set_trace()
        self.dzi_obj = None

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.__filename)

    @classmethod
    def detect_format(cls, filename):
        return kfb_lowlevel.detect_vendor(filename)

    def close(self):
        kfb_lowlevel.kfbslide_close(self._osr)

    @property
    def level_count(self):
        return kfb_lowlevel.kfbslide_get_level_count(self._osr)
    
    @property
    def level_dimensions(self):
        return tuple(kfb_lowlevel.kfbslide_get_level_dimensions(self._osr, i)
                     for i in range(self.level_count))
    @property
    def dimensions(self):
        """A (width, height) tuple for level 0 of the image."""
        return self.level_dimensions[0]
    @property
    def level_downsamples(self):
        return tuple(kfb_lowlevel.kfbslide_get_level_downsample( self._osr, i)
                     for i in range(self.level_count))

    @property
    def properties(self):
        return _KfbPropertyMap(self._osr)

    @property
    def associated_images(self):
        return _AssociatedImageMap(self._osr)

    def get_best_level_for_downsample(self, downsample):
        return  kfb_lowlevel.kfbslide_get_best_level_for_downsample(self._osr, downsample)

    def _fixed_read_region(self, location, level, size=(256, 256)):
        #import pdb
        x = int(location[0])
        y = int(location[1])
        # img_index = kfbRef.img_count
        kfbRef.img_count += 1
        
        #print("img_index : ", img_index, "Level : ", level, "Location : ", x , y)
        region_ =  kfb_lowlevel.kfbslide_read_region(self._osr, level, x, y)

        return region_

    def get_dzi(self):
        if self.dzi_obj is None:
            self.dzi_obj = KfbDeepZoomGenerator(self)
        return self.dzi_obj
        
    def read_region(self, location=[0,0], level=0, size = [256, 256] ):
        '''
        Parameters:
        ------------
            locatation: (x, y) at level 0
            size:  (x, y)
        Return:
        ------------
        
        '''
        TILE_SIZE = 256
        r_loc_level0 = list(location)
        r_size = list(size)
        
        dzi_obj = self.get_dzi()
        
        downsample_ratio = self.level_downsamples[level]

        r_loc = [int(r/downsample_ratio) for r in r_loc_level0]

        tile_index = dzi_obj._dz_levels - 1 - level
        x_count, y_count = dzi_obj._t_dimensions[tile_index]
        # substract 1 to crop boundary
        x_count, y_count = x_count - 1, y_count - 1
        x_dim, y_dim = dzi_obj._z_dimensions[tile_index]
        assert x_count*256 <= x_dim and y_count*256 <= y_dim

        # Get shift to make up for 256
        start_shift = [loc % TILE_SIZE for loc in r_loc]
        # calculate 256 integral size
        p_loc  =  list(map(operator.sub, r_loc,  start_shift))
        t_size =  list(map(operator.add, r_size, start_shift))
        # add end_shift to make up for 256 in the end part
        end_shift = [TILE_SIZE - loc % TILE_SIZE for loc in t_size]
        p_size = list(map(operator.add, t_size, end_shift))

        region_img = np.zeros((p_size[1], p_size[0], 3), dtype=np.uint8) + 1
        
        num_x, num_y = int(p_size[0]/TILE_SIZE), int(p_size[1]/TILE_SIZE)

        start_x, start_y = math.floor(p_loc[0]/TILE_SIZE), math.floor(p_loc[1]/TILE_SIZE)
        
        # read region one by one
        for rx in range(num_x): # traverse through x
            for ry in range(num_y): # traverse though y
                #args, z_size = self._get_tile_info(level, location=[p_loc[0]+rx*TILE_SIZE, p_loc[1]+ry*TILE_SIZE],
                #                                      )
                #cur_region = self._fixed_read_region((p_loc[0]+rx*TILE_SIZE, p_loc[1]+ry*TILE_SIZE), 
                #                                      level=level)
                cur_region = dzi_obj.get_tile(tile_index, (rx+start_x, ry+start_y))
                if isinstance(cur_region, np.ndarray):
                    buf = BytesIO(cur_region)
                    cur_region = Image.open(buf)
                    cur_region = cur_region.convert("RGB")
                # pdb.set_trace()
                cur_img = np.asanyarray(cur_region).astype(np.uint8)
                #region_img[ry*TILE_SIZE:(ry+1)*TILE_SIZE, rx*TILE_SIZE:(rx+1)*TILE_SIZE, :] = cur_img
                _rsize, _csize, _ch = cur_img.shape
                region_img[ry*TILE_SIZE:ry*TILE_SIZE+_rsize, rx*TILE_SIZE:rx*TILE_SIZE+_csize, :] = cur_img
                

                # cur_region_io.close()

                cur_img = None
                # cur_region_io = None
                buf = None
                cur_region = None
        
        roi_img = region_img[start_shift[1]:start_shift[1]+ r_size[1],
                             start_shift[0]:start_shift[0] + r_size[0], 0:3]
        pil_image = Image.fromarray(roi_img)

        return pil_image


    # def __read_region__(self, location=[0,0], level=0, size = [256, 256] ):
    #     '''
    #     Parameters:
    #     ------------
    #         locatation: (x, y)
    #         size:  (x, y)
    #     Return:
    #     ------------
    #     don't use this function, it's bad
    #     '''
    #     TILE_SIZE = 256
    #     r_loc = list(location)
    #     r_size = list(size)
    #
    #     # Get shift to make up for 256
    #     start_shift = [loc % TILE_SIZE for loc in r_loc]
    #     # calculate 256 integral size
    #     p_loc  =  list(map(operator.sub, r_loc,  start_shift))
    #     t_size =  list(map(operator.add, r_size, start_shift))
    #     # add end_shift to make up for 256 in the end part
    #     end_shift = [TILE_SIZE - loc % TILE_SIZE for loc in t_size]
    #     p_size = list(map(operator.add, t_size, end_shift))
    #
    #     region_img = np.zeros((p_size[1], p_size[0], 3), dtype=np.uint8) + 1
    #
    #     num_x, num_y = int(p_size[0]/TILE_SIZE), int(p_size[1]/TILE_SIZE)
    #     # read region one by one
    #     for rx in range(num_x): # traverse through x
    #         for ry in range(num_y): # traverse though y
    #             #args, z_size = self._get_tile_info(level, location=[p_loc[0]+rx*TILE_SIZE, p_loc[1]+ry*TILE_SIZE],
    #             #                                      )
    #             cur_region = self._fixed_read_region((p_loc[0]+rx*TILE_SIZE, p_loc[1]+ry*TILE_SIZE),
    #                                                   level=level)
    #             if isinstance(cur_region, np.ndarray):
    #                 buf = BytesIO(cur_region)
    #                 cur_region_io = Image.open(buf)
    #             cur_region = cur_region_io.convert("RGB")
    #             cur_img = np.asanyarray(cur_region).astype(np.uint8)
    #             region_img[ry*TILE_SIZE:(ry+1)*TILE_SIZE, rx*TILE_SIZE:(rx+1)*TILE_SIZE, :] = cur_img
    #             cur_region_io.close()
    #
    #             cur_img = None
    #             cur_region_io = None
    #             buf = None
    #             cur_region = None
    #
    #     roi_img = region_img[start_shift[1]:start_shift[1]+ r_size[1],
    #                          start_shift[0]:start_shift[0] + r_size[0], 0:3]
    #     return roi_img




class _KfbPropertyMap(Mapping):
    def _keys(self):
        return kfb_lowlevel.kfbslide_property_names(self._osr)

    def __getitem__(self, key):
        v = kfb_lowlevel.kfbslide_property_value( self._osr, key)
        if v is None:
            raise KeyError()
        return v
    def __init__(self, osr):
        self._osr = osr

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, dict(self))

    def __len__(self):
        return len(self._keys())

    def __iter__(self):
        return iter(self._keys())



class _AssociatedImageMap():

    def __init__(self, osr):
        self._osr = osr

    def _keys(self):
        return kfb_lowlevel.kfbslide_get_associated_image_names(self._osr)

    def __getitem__(self, key):
        if key not in self._keys():
            raise KeyError()
        return kfb_lowlevel.kfbslide_read_associated_image(self._osr, key)

def open_kfbslide(filename):
    try:
        return KfbSlide(filename)
    except:
        try:
            return KfbSlide(filename.encode('gbk'))
        except Exception:
            try:
                return KfbSlide(filename.encode('utf-8'))
            except:
                return None

