from __future__ import division
from io import BytesIO
import math
from PIL import Image
from xml.etree.ElementTree import ElementTree, Element, SubElement
import operator
import numpy as np
PROPERTY_NAME_BACKGROUND_COLOR = u'openslide.background-color'
PROPERTY_NAME_BOUNDS_X         = u'openslide.bounds-x'
PROPERTY_NAME_BOUNDS_Y         = u'openslide.bounds-y'
PROPERTY_NAME_BOUNDS_WIDTH     = u'openslide.bounds-width'
PROPERTY_NAME_BOUNDS_HEIGHT    = u'openslide.bounds-height'

class KfbDeepZoomGenerator( object ):

    BOUNDS_OFFSET_PROPS = (PROPERTY_NAME_BOUNDS_X,
                           PROPERTY_NAME_BOUNDS_Y)
    BOUNDS_SIZE_PROPS   = (PROPERTY_NAME_BOUNDS_WIDTH,
                           PROPERTY_NAME_BOUNDS_HEIGHT)
    def __init__(self, osr, tile_size = 256, overlap=1, limit_bounds = False):
        #import pdb
        #pdb.set_trace()
        tile_size = 256
        overlap = 0
        self._osr = osr
        self._z_t_downsample = tile_size #tile_size
        self._z_overlap = overlap #overlap
        self._limit_bounds = limit_bounds

        self._l_dimensions = osr.level_dimensions
        self._l0_offset = ( 0, 0)
        self._l0_dimensions = self._l_dimensions[0]

        z_size = self._l0_dimensions
        z_dimensions = [z_size]
        while z_size[0] > 1 or z_size[1] > 1:
            z_size = tuple( max( 1, int(math.ceil( z/2))) for z in z_size)
            z_dimensions.append(z_size)
        self._z_dimensions = tuple(reversed(z_dimensions))

        #Tile
        tiles = lambda z_lim: int(math.ceil(z_lim / self._z_t_downsample))
        self._t_dimensions = tuple((tiles(z_w), tiles(z_h))
                                   for z_w, z_h in self._z_dimensions)

        # Deep Zoom level count
        self._dz_levels = len(self._z_dimensions)

        # Total downsamples for each Deep Zoom level
        l0_z_downsamples = tuple( 2 ** ( self._dz_levels - dz_level - 1)
                                  for dz_level in range(self._dz_levels))

        # Preferred slide levels for each Deep Zoom level
        self._slide_from_dz_level = tuple(
            self._osr.get_best_level_for_downsample(d)
            for d in l0_z_downsamples)

        # Piecewise downsamples
        self._l0_l_downsamples = self._osr.level_downsamples
        self._l_z_downsamples = tuple(
            l0_z_downsamples[dz_level] /
            self._l0_l_downsamples[ self._slide_from_dz_level[dz_level]]
            for dz_level in range(self._dz_levels))

        # Slide background color
        self._bg_color = '#' + self._osr.properties.get(
            PROPERTY_NAME_BACKGROUND_COLOR, 'ffffff')

    def __repr__(self):
        return "%s(%r, tile_size=%r, overlap=%r, limit_bounds=%r)" % (
            self.__class__.__name__, self._osr, self._z_t_downsample,
            self._z_overlap, self._limit_bounds)

    @property
    def level_count(self):
        return self._dz_levels

    @property
    def level_tiles(self):
        return self._t_dimensions

    @property
    def level_dimensions(self):
        return self._z_dimensions

    @property
    def tile_count(self):
        return sum(t_cols * t_rows for t_cols, t_rows in self._t_dimensions)
    '''
    def get_tile(self, level, address):
        tile = self._osr.read_region( address, level, (256, 256))
        from io import BytesIO
        buf = BytesIO(tile)
        img = Image.open(buf)
        return img
    '''

    def get_tile(self, level, address):
        """ Return an RGB PIL.Image for a tile (which is 256 x 256).
            level:     the Deep Zoom level.
            address:   the address of the tile within the level as a (col, row)
                       tuple. 
        """
        # Read tile
        args, z_size = self._get_tile_info(level, address)
        tile = self._osr._fixed_read_region(*args)
        '''
        # Apply on solid background
        bg = Image.new('RGB', tile.size, self._bg_color)
        tile = Image.composite(tile, bg, tile)

        # Scale to the correct size
        if tile.size != z_size:
            tile.thumbnail(z_size, Image.ANTIALIAS)
        '''
        from io import BytesIO
        buf  = BytesIO(tile)
        tile = Image.open(buf)
        tile = tile.convert("RGB")

        return tile

    def _get_tile_info(self, dz_level, t_location):
        # Check parameters
        if dz_level < 0 or dz_level >= self._dz_levels:
            raise ValueError("Invalid level")
        for t, t_lim in zip(t_location, self._t_dimensions[dz_level]):
            if t < 0 or t >= t_lim:
                raise ValueError("Invalid address")

        # Get preferred slide level
        slide_level = self._slide_from_dz_level[dz_level]

        # Calculate top/left and bottom/right overlap
        z_overlap_tl = tuple(self._z_overlap * int(t != 0)
                    for t in t_location)
        z_overlap_br = tuple(self._z_overlap * int(t != t_lim - 1)
                    for t, t_lim in
                    zip(t_location, self.level_tiles[dz_level]))

        # Get final size of the tile
        z_size = tuple(min(self._z_t_downsample,
                    z_lim - self._z_t_downsample * t) + z_tl + z_br
                    for t, z_lim, z_tl, z_br in
                    zip(t_location, self._z_dimensions[dz_level],
                    z_overlap_tl, z_overlap_br))

        # Obtain the region coordinates
        z_location = [self._z_from_t(t) for t in t_location]
        l_location = [self._l_from_z(dz_level, z - z_tl)
                    for z, z_tl in zip(z_location, z_overlap_tl)]
        # Round location down and size up, and add offset of active area
        l0_location = tuple(int(self._l0_from_l(slide_level, l) + l0_off)
                    for l, l0_off in zip(l_location, self._l0_offset))
        l_size = tuple(int(min(math.ceil(self._l_from_z(dz_level, dz)),
                    l_lim - math.ceil(l)))
                    for l, dz, l_lim in
                    zip(l_location, z_size, self._l_dimensions[slide_level]))

        # Return read_region() parameters plus tile size for final scaling
        return ((l0_location, slide_level, l_size), z_size)


    def load_kfb2arr(kfb_filepath, level=0, wsi_dim=None):
        kfb_slide = kfb_deepzoom.KfbDeepZoomGenerator(kfbslide.KfbSlide(kfb_filepath.encode('gbk')))
        tile_index = kfb_slide._dz_levels - 1 - level
        x_count, y_count = kfb_slide._t_dimensions[tile_index]
        # substract 1 to crop boundary
        x_count, y_count = x_count - 1, y_count - 1
        x_dim, y_dim = kfb_slide._z_dimensions[tile_index]
        assert x_count*256 <= x_dim and y_count*256 <= y_dim

        wsi_img = np.zeros((y_count*256, x_count*256, 3)) # Crop boundary
        for index_x in range(x_count):
            for index_y in range(y_count):
                start_x, start_y = index_x*256, index_y*256
                wsi_img[start_y:start_y+256, start_x:start_x+256, :] = kfb_slide.get_tile(tile_index, (index_x, index_y))
        wsi_img = wsi_img / 255.0
        # Select regions
        if wsi_dim != None:
            wsi_img = wsi_img[:wsi_dim[0], :wsi_dim[1], :]

        return wsi_img
        
    def read_region(self, location=[0,0], level=0, size = [256, 256]):
        '''
        Parameters:
        ------------
            locatation: (x, y)
            size:  (x, y)
        Return:
        ------------
        don't use this function, it's bad
        '''
        TILE_SIZE = 256
        r_loc = list(location)
        r_size = list(size)

        
        tile_index = self._dz_levels - 1 - level
        x_count, y_count = self._t_dimensions[tile_index]
        # substract 1 to crop boundary
        x_count, y_count = x_count - 1, y_count - 1
        x_dim, y_dim = self._z_dimensions[tile_index]
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
        # read region one by one
        for rx in range(num_x): # traverse through x
            for ry in range(num_y): # traverse though y
                #args, z_size = self._get_tile_info(level, location=[p_loc[0]+rx*TILE_SIZE, p_loc[1]+ry*TILE_SIZE],
                #                                      )
                #cur_region = self._fixed_read_region((p_loc[0]+rx*TILE_SIZE, p_loc[1]+ry*TILE_SIZE), 
                #                                      level=level)
                cur_region = kfb_slide.get_tile(tile_index, (rx, ry))

                if isinstance(cur_region, np.ndarray):
                    buf = BytesIO(cur_region)
                    cur_region_io = Image.open(buf)
                cur_region = cur_region_io.convert("RGB")
                cur_img = np.asanyarray(cur_region).astype(np.uint8)
                region_img[ry*TILE_SIZE:(ry+1)*TILE_SIZE, rx*TILE_SIZE:(rx+1)*TILE_SIZE, :] = cur_img
                cur_region_io.close()

                cur_img = None
                cur_region_io = None
                buf = None
                cur_region = None
        
        roi_img = region_img[start_shift[1]:start_shift[1]+ r_size[1],
                             start_shift[0]:start_shift[0] + r_size[0], 0:3]
        return roi_img

    def _l0_from_l(self, slide_level, l):
        return self._l0_l_downsamples[slide_level] * l

    def _l_from_z(self, dz_level, z):
        return self._l_z_downsamples[dz_level] * z

    def _z_from_t(self, t):
        return self._z_t_downsample * t

    def get_dzi(self, format = "jpeg"):
        """Return a string containing the XML metadata for the .dzi file.

                format:    the format of the individual tiles ('png' or 'jpeg')"""
        image = Element('Image', TileSize=str(self._z_t_downsample),
                        Overlap=str(self._z_overlap), Format="jpeg",
                        xmlns='http://schemas.microsoft.com/deepzoom/2008')
        w, h = self._l0_dimensions
        SubElement(image, 'Size', Width=str(w), Height=str(h))
        tree = ElementTree(element=image)
        buf = BytesIO()
        tree.write(buf, encoding='UTF-8')
        return buf.getvalue().decode('UTF-8')

