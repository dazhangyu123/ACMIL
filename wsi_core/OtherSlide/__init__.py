#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os.path,math,sys
dir = os.path.dirname( os.path.abspath(__file__))


os.environ["PATH"] = dir + "\\openslidec;" + os.environ["PATH"]

from PIL import Image
import cv2
import numpy as np
import json

from ..SlideBase import SlideBase
from . import openslide
from .openslide.deepzoom import DeepZoomGenerator

Image.MAX_IMAGE_PIXELS = None


class OtherSlide(SlideBase):
    def __init__(self,filename):
        # print("OpenSlide")
        self.filename = filename
        if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', 'png', '.bmp']:
            try:
                self.slide = openslide.ImageSlide(filename)
            except:
                self.slide=  openslide.OpenSlide(filename)
        else:
            self.slide = openslide.open_slide(filename)
        self.width, self.height = self.slide.dimensions
        SlideBase.__init__(self)
        self.getTile( 0,0 , self.maxlvl )#初始化完成后，才释放lock

    def read(self, location=[0,0], size=None, scale=1.0, greyscale=False):
        '''
        :param location: (x, y) at level=0
        :param size: (width, height)
        :param scale: resize scale, scale>1 -> zoom out, scale<1 -> zoom in
        :param greyscale: if True, convert image to greyscale
        :return: a numpy image,  np_img.shape=[height, width, channel=1 or 3]
        '''
        if size == None:
            width, height = self.width, self.height
        else:
            width, height = size

        crop_start_x, crop_start_y = location
        crop_level = self.slide.get_best_level_for_downsample(scale)
        resize_ratio = self.slide.level_downsamples[crop_level]/scale

        # make sure the crop region is inside the slide
        crop_start_x = math.ceil(min(max(crop_start_x, 0), self.width))
        crop_start_y = math.ceil(min(max(crop_start_y, 0), self.height))
        crop_end_x = math.ceil(min(max(width+crop_start_x, 0), self.width))
        crop_end_y = math.ceil(min(max(height+crop_start_y, 0), self.height))

        crop_width = math.ceil((crop_end_x - crop_start_x)/self.slide.level_downsamples[crop_level])
        crop_height = math.ceil((crop_end_y - crop_start_y)/self.slide.level_downsamples[crop_level])

        if crop_height == 0 or crop_width == 0:
            return None


        # print(crop_start_x, crop_start_y)
        # print(crop_level)
        # print(crop_width, crop_height)

        crop_region = self.slide.read_region(
            location=(crop_start_x, crop_start_y),
            level=crop_level,
            size=(crop_width, crop_height)
        )

        crop_region = np.array(crop_region)[:, :, 0:3]

        if greyscale:
            crop_region = 0.2989*crop_region[:,:,0] + 0.5870*crop_region[:,:,1] + 0.1140*crop_region[:,:,2]
            crop_region = crop_region[:,:,np.newaxis]

        crop_region = cv2.resize(crop_region, (math.ceil(crop_width*resize_ratio), math.ceil(crop_height*resize_ratio)))
        return crop_region


    def getTile(self,x,y,z):
        dzi_obj = DeepZoomGenerator(self.slide, tile_size=1024, overlap=0)
        # print(x,y,z)
        return dzi_obj.get_tile(level=z, address=(x,y))


    def saveLabel(self,path):
        try:
            self.slide.associated_images["label"].save(path)
        except:
            pass


    @property
    def mpp(self):
        mpp = None
        try:
            slide_properties = self.slide.properties
            if 'openslide.mpp-x' in slide_properties:
                mpp = float(slide_properties['openslide.mpp-x'])
                return mpp
            if 'tiff.XResolution' in slide_properties:
                unit = slide_properties['tiff.ResolutionUnit']
                if unit == 'centimeter':
                    return 10000 / float(slide_properties['tiff.XResolution'])
                if unit == 'inch':
                    return 25400 / float(slide_properties['tiff.XResolution'])
 
            with open(os.path.join(os.path.dirname(self.filename), "index.json"), "r", encoding="utf-8") as f:
                slide_info = json.load(f)
                mpp = slide_info.get("mppx")
                if mpp is not None:
                    return float(mpp)
        except:
            pass

        return mpp

