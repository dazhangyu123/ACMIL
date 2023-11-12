import numpy as np
from skimage import draw

def getBorder(slide, x_coords=[], y_coords=[]):
    if (x_coords is None or y_coords is None) or (len(x_coords) < 3 or len(y_coords) < 3):
        ymin, xmin = 0, 0
        height, width = slide.height, slide.width
        ymax, xmax = height, width
    else:
        ymin, xmin, ymax, xmax = min(max(0, min(y_coords)), slide.height), \
                                 min(slide.width, max(0, min(x_coords))), \
                                 min(slide.height, max(0, max(y_coords))), \
                                 min(slide.width, max(0, max(x_coords)))
        height = ymax - ymin
        width = xmax - xmin

    return xmin, ymin, xmax, ymax, height, width


def getMask(slide, x_coords=[], y_coords=[]):
    if (x_coords is None or y_coords is None) or (len(x_coords) < 3 or len(y_coords) < 3):
        return np.ones((slide.height, slide.width))
    else:
        ymin, xmin, ymax, xmax = min(max(0, min(y_coords)), slide.height), \
                                 min(slide.width, max(0, min(x_coords))), \
                                 min(slide.height, max(0, max(y_coords))), \
                                 min(slide.width, max(0, max(x_coords)))
        height = ymax - ymin
        width = xmax - xmin

        mask = np.zeros((height, width))
        y_coords = [y - ymin for y in y_coords]
        x_coords = [x - xmin for x in x_coords]
        rr, cc = draw.polygon(y_coords, x_coords)
        cc = np.clip(cc, 0, width - 1).astype(int)
        rr = np.clip(rr, 0, height - 1).astype(int)
        mask[rr, cc] = 1
        return mask