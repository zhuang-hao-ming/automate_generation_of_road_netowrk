# -*- coding: utf-8 -*-



'''

将矢量的轨迹栅格化

'''

import geopandas as gpd
import numpy as np
from skimage.morphology import skeletonize
import cv2
import rasterio
import skimage.io


profile = {
    'driver': 'GTiff',
    'dtype': 'uint16',
    'nodata': 65535,
    'width': None,
    'height': None,
    'count': 1,
    'crs': None,
    'transform': None,
    'blockxsize': None,
    'blockysize': None
    }



def get_extent(shp_file='./shp/trj.shp'):
    df = gpd.read_file(shp_file)

    x_list = []
    y_list = []
    for idx, row in df.iterrows():
        for x, y in row['geometry'].coords:
            x_list.append(x)
            y_list.append(y)

    min_x = min(x_list)
    max_x = max(x_list)
    min_y = min(y_list)
    max_y = max(y_list)

    return min_x, min_y, max_x, max_y


def rasterize(shp_file='./shp/trj.shp', output_raster_name='sk.tif'):
    min_x, min_y, max_x, max_y = get_extent(shp_file)
    cell_size = 2
    min_x, min_y, max_x, max_y = min_x-cell_size, min_y-cell_size, max_x+cell_size, max_y+cell_size
    

    width = int((max_x - min_x) / cell_size)
    height = int((max_y - min_y) / cell_size)

    density_img = np.zeros((height, width), dtype=np.uint16)
    trip_list = gpd.read_file(shp_file)['geometry']
    
    for trip in trip_list:
        temp = np.zeros((height, width), dtype=np.uint16)

        trip = trip.coords[:]
        for (orig, dest) in zip(trip, trip[1:]):
            orig_x, orig_y = orig
            dest_x, dest_y = dest

            ox = int((orig_x - min_x) / cell_size)
            oy = height - int((orig_y - min_y) / cell_size) - 1

            dx = int((dest_x - min_x) / cell_size)
            dy = height - int((dest_y - min_y) / cell_size) - 1

            cv2.line(temp,(ox,oy),(dx,dy),1,1,cv2.LINE_AA)

        density_img += temp

    # skimage.io.imsave('./tif/density.tif', density_img)
    blur = cv2.GaussianBlur(density_img, (17,17), 0)
    
    min_val = np.min(blur)
    max_val = np.max(blur)
    mask = (blur - min_val) / (max_val - min_val) * 255
    # skimage.io.imsave('./tif/mask.tif', mask)
    ret, threshold = cv2.threshold(mask, 10, 1, cv2.THRESH_BINARY)
    threshold = threshold.astype(np.uint8)

    # skimage.io.imsave('./tif/threshold_1.tif', threshold)

    skeleton_img = skeletonize(threshold)
    skeleton_img = skeleton_img.astype(np.uint16)
    profile.update({
        'width': width,
        'height': height,
        'crs': trip_list.crs,
        'transform': (cell_size, 0.0, min_x, 0.0, -cell_size, max_y),
        'blockxsize': cell_size,
        'blockysize': cell_size
    })

    
    
    with rasterio.open(output_raster_name, 'w', **profile) as dst:
        dst.write(skeleton_img, 1)


if __name__ == '__main__':
    rasterize(shp_file='./shp/trj.shp', output_raster_name='./tif/skeleton.tif')