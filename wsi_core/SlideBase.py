import math
import numpy as np
from PIL import Image
from skimage import draw

class SlideBase:
    def __init__(self):

        mx = max(self.width,self.height)
        self.maxlvl = math.ceil(math.log2(mx))


    #虚函数，每个继承类必须重载。
    def read(self, location=[0,0], size=None, scale=1.0, greyscale=False):
        '''
        :param location: (x, y) at level=0
        :param size: (width, height)
        :param scale:  downsampling ratio
        :param greyscale: if True, convert image to greyscale
        :return: a numpy image,  np_img.shape=[height/scale, width/scale, channel=1 or 3]
        '''
        print("虚函数，每个继承类必须重载。")
        print("x,y为原始图上的x，y")
        print("w,h为宽度高度 ")
        print("scale为要缩小多少倍")
        print("返回内存阵列")

    def getTile(self,x,y,z):
        scale = math.pow(2, self.maxlvl - z)
        r = 1024*scale
        tile = self.read([x * r, y * r], [r , r], scale, greyscale=False)

        return Image.fromarray(tile,mode='RGB')


    def getWindow(self, xindex, yindex, window_size=[100,100], overlap=[50,50], scale=1, padding=True, bbox=None):
        if bbox is None:
            x_min, y_min, x_max, y_max = 0, 0, self.width, self.height
        else:
            x_min, y_min, x_max, y_max = bbox

        window_w, window_h = window_size
        overlap_w, overlap_h = overlap

        window_w*=scale
        window_h*=scale
        overlap_w*=scale
        overlap_h*=scale

        stride_w, stride_h = window_w-overlap_w, window_h-overlap_h


        crop_start_x = x_min + xindex*stride_w
        crop_start_y = y_min + yindex*stride_h

        # crop_size_w = window_w * scale
        # crop_size_h = window_h * scale

        img = self.read([crop_start_x, crop_start_y], [window_w, window_h], scale)
        if padding:
            img =pad_img(img, window_size)

        return img


    def get_slide_window_info(self, standard_mpp, window_size=[512,512], overlap=[128,128], cut_leftover=[0,0], bbox=None):
        '''
        # compute resize scale, number of rows and columns for sliding window
        :param standard_mpp: mpp of training dataset
        :param window_size: slide window size, order is in [width, height]
        :param overlap: overlaps between two adjacent window, order is in [width, height]
        :param cut_leftover: omit the leftover if leftover <= cut_leftover,  [width, height]
        :param bbox: box area to run the slide window, order is in [x_min, y_min, x_max, y_max]
        :return:
                scale: dowmsampling ratio
                (num_x, num_y):
                    num_x: number of windows in horizontal direction
                    num_y: number of windows in vertical direction
        '''

        if standard_mpp is None or self.mpp is None:
            scale = 1
        else:
            scale = standard_mpp/self.mpp

        if bbox is None:
            x_min, y_min, x_max, y_max = 0, 0, self.width, self.height
        else:
            x_min, y_min, x_max, y_max = bbox

        height, width = y_max - y_min, x_max - x_min

        window_w, window_h = window_size
        overlap_w, overlap_h = overlap
        cut_leftover_w, cut_leftover_h = cut_leftover

        window_w        *=scale
        window_h        *=scale
        overlap_w       *=scale
        overlap_h       *=scale
        cut_leftover_h  *=scale
        cut_leftover_w  *=scale

        stride_h, stride_w = window_h-overlap_h, window_w-overlap_w

        #Width = window_w + (n-1)*stride_w + leftover_w
        #Height = window_h + (n-1)*stride_h + leftover_h
        num_x, num_y = 1 + math.floor((width - window_w) / stride_w), 1 + math.floor((height - window_h) / stride_h)
        num_x, num_y = max(num_x, 1), max(num_y, 1)

        leftover_w = width - window_w - (num_x-1)*stride_w
        leftover_h = height - window_h - (num_y-1)*stride_h


        if leftover_w > cut_leftover_w:
            num_x += 1
        if leftover_h > cut_leftover_h:
            num_y += 1

        return scale, (num_x, num_y)


    def getThumbnail(self, size=500):
        '''
        :param size: thumbnail image size
        :return:  a thumbnail image
        '''
        thumbnail_img = None
        try:
            thumbnail_img = self.slide.get_thumbnail(size=(size, size))
        except:
            try:
                for k, v in self.slide.associated_images.items():
                    if 'thumbnail' in str(k):
                        thumbnail_img = self.slide.associated_images[k]
            except:
                maxSize = max(self.height, self.width)
                scale_ratio = maxSize / size
                np_thumb = self.read(location=[0, 0], size=[self.width, self.height], scale=scale_ratio)
                thumbnail_img = Image.fromarray(np_thumb, mode='RGB')

        if thumbnail_img:
            if thumbnail_img.mode == 'RGBA':
                thumbnail_img = thumbnail_img.convert('RGB')


        return thumbnail_img

    def saveLabel(self,path):
        pass


    def get_border(self, x_coords=[], y_coords=[]):
        if (x_coords is None or y_coords is None) or (len(x_coords) < 3 or len(y_coords) < 3):
            ymin, xmin = 0, 0
            height, width = self.height, self.width
            ymax, xmax = height, width
        else:
            ymin, xmin, ymax, xmax = min(max(0, min(y_coords)), self.height), \
                                     min(self.width, max(0, min(x_coords))), \
                                     min(self.height, max(0, max(y_coords))), \
                                     min(self.width, max(0, max(x_coords)))
            height = ymax - ymin
            width = xmax - xmin

        return int(xmin), int(ymin), int(xmax), int(ymax), int(height), int(width)


    def get_mask(self, x_coords=[], y_coords=[]):
        if (x_coords is None or y_coords is None) or (len(x_coords) < 3 or len(y_coords) < 3):
            return np.ones((self.height, self.width))
        else:
            ymin, xmin, ymax, xmax = min(max(0, min(y_coords)), self.height), \
                                     min(self.width, max(0, min(x_coords))), \
                                     min(self.height, max(0, max(y_coords))), \
                                     min(self.width, max(0, max(x_coords)))
            height = ymax - ymin
            width = xmax - xmin

            mask = np.zeros((height, width))
            y_coords = [y - ymin for y in y_coords]
            x_coords = [x - xmin for x in x_coords]
            rr, cc = draw.polygon(y_coords, x_coords)
            cc = np.clip(cc, 0, width - 1).astype(int)
            rr = np.clip(rr, 0, height - 1).astype(int)
            mask[rr, cc]= 1
            return mask


def pad_img(img, pad_size=(512,512)):
    if img.shape[0:2] == pad_size:
        return img
    else:
        new_img = np.zeros((pad_size[0], pad_size[1], img.shape[2]))
        new_img[:img.shape[0], :img.shape[1],:] = img
        return new_img