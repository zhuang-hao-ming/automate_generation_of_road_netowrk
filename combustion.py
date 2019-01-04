# -*- coding: utf-8 -*-


import skimage.io
import numpy as np
from collections import deque
from shapely.geometry import LineString
import geopandas as gpd
import rasterio

class MainCrossing():
    def __init__(self, crossing_stack):
        self.row = 0
        self.col = 0
        self.crossing_stack = crossing_stack
        for (i, j) in crossing_stack:
            self.row += i
            self.col += j
        self.row /= float(len(crossing_stack))
        self.col /= float(len(crossing_stack))

    @property
    def location(self):
        return (self.row, self.col)

class Node():
    def __init__(self, x, y):
        self.x = x
        self.y = y



def raw_combustion(sub_img, seed):
    combusting = deque()

    sub_img[seed] = 0
    combusting.append(seed)
    while len(combusting) > 0:
        top_row, top_col = combusting.popleft()

        for row in range(top_row-1, top_row+2):
            for col in range(top_col-1, top_col+2):
                if row < 0 or row > 4 or col < 0 or col > 4:
                    continue
                if sub_img[row, col] == 1:
                    sub_img[row, col] = 0
                    combusting.append((row, col))

    return sub_img

def combustion(sub_img):

    '''

    '''


    combusting = deque()

    sub_img[2, 2] = 0
    combusting.append((2,2))


    resore_cell = []
    while len(combusting) > 0:
        top_row, top_col = combusting.popleft()
        if not (top_row > 0 and top_row < 4 and top_col > 0 and top_col < 4):
            resore_cell.append((top_row, top_col))
        for row in range(top_row-1, top_row+2):
            for col in range(top_col-1, top_col+2):
                if row < 0 or row > 4 or col < 0 or col > 4:
                    continue
                if sub_img[row, col] == 1:
                    sub_img[row, col] = 0
                    combusting.append((row, col))

    
    fringe = np.zeros((5,5), dtype=np.int)
    for cell in resore_cell:
        fringe[cell] = 1
    

    implement_count = 0
    while True:
        val_one_cells = np.where(fringe==1)
        
        
        if len(val_one_cells[0]) == 0:
            break
        implement_count += 1
        seed = (val_one_cells[0][0], val_one_cells[1][0])
        # print(seed)
        fringe = raw_combustion(fringe, seed)
    # print(implement_count)
    return implement_count



def find_crossing_point(skeleton_file):
    '''
    对骨架栅格文件进行处理，找到crossing pixel
    '''
    skeleton = skimage.io.imread(skeleton_file)
    cross_img = skeleton.copy()

    border_pixel_list = np.where(skeleton==1)
    # print('there are {} border pixels'.format(len(border_pixel_list[0])))

    
    cnt = 0
    for i, j in zip(*border_pixel_list):
        
        if (i-2) < 0 or (j-2) < 0 or (i+3) > skeleton.shape[0] or (j+3) > skeleton.shape[1]:
            print('border condition --')
            continue

        sub_img = skeleton[i-2:i+3, j-2:j+3].copy()

        implement_count = combustion(sub_img)
        if implement_count == 2:
            cross_img[i, j] = 1
        elif implement_count == 0:
            cross_img[i, j] = 0
        elif implement_count == 1 or implement_count > 2:
            cross_img[i, j] = 2

        
        cnt += 1
        # if cnt % 100 == 0:
        #     print('cnt:', cnt)

    # skimage.io.imsave('./cross.tif', cross_img)
    return cross_img
    


def find_main_crossing(cross_img):
    '''
    对crossing进行处理， 给组成同一个交叉口的crossing赋值同一个main crossing
    '''
    
    crossing_main_dict = {}
    cross_pixel_list = np.where(cross_img==2)

    for i,j in zip(cross_pixel_list[0], cross_pixel_list[1]):

        crossing_stack = []
        combusting_queue = deque([])

        if cross_img[i,j] == 2:
            cross_img[i,j] = 3
            combusting_queue.appendleft((i, j))
        else:
            if (i, j) not in crossing_main_dict:
                assert(False)
        
        while (len(combusting_queue) > 0):
            top_row, top_col = combusting_queue.popleft()
            crossing_stack.append((top_row, top_col))
            for row in range(top_row-1, top_row+2):
                for col in range(top_col-1, top_col+2):
                    if row < 0 or row >= cross_img.shape[0] or col < 0 or col >= cross_img.shape[1]:
                        continue
                    if cross_img[row, col] == 2:
                        cross_img[row, col] = 3
                        combusting_queue.append((row, col))

        if len(crossing_stack) > 0:
            new_main_crossing = MainCrossing(crossing_stack)
            for crossing in crossing_stack:
                crossing_main_dict[crossing] = new_main_crossing

    # skimage.io.imsave('./main_cross.tif', cross_img)
    return cross_img, crossing_main_dict

def find_edges(cross_img):
    '''
    从main crossing出发找到连接每一个crossing的边
    '''

    segment_list = []
    cross_pixel_list = np.where(cross_img == 3)

    # print('len of crossing: {}'.format(len(cross_pixel_list[0])))

    for (i, j) in zip(cross_pixel_list[0], cross_pixel_list[1]):
        
        combusting_queue = deque([])
        edge_nodes = []

        combusting_queue.append((i, j))

        while len(combusting_queue) > 0:
            top_row, top_col = combusting_queue.popleft()
            edge_nodes.append((top_row, top_col))

            for row in range(top_row-1, top_row+2):
                for col in range(top_col-1, top_col+2):
                    if row < 0 or row >= cross_img.shape[0] or col < 0 or col >= cross_img.shape[1]:
                        continue
                    if cross_img[row, col] == 1:
                        cross_img[row, col] = 0
                        combusting_queue.append((row, col))

        if len(edge_nodes) == 1:
            continue

        end_row, end_col = edge_nodes[-1]
        for row in range(end_row-1, end_row+2):
            for col in range(end_col-1, end_col+2):
                if row < 0 or row >= cross_img.shape[0] or col < 0 or col >= cross_img.shape[1]:
                    continue
                if cross_img[row, col] == 3:
                    edge_nodes.append((row, col))
            
        assert(cross_img[edge_nodes[-1]] == 3)
        assert(cross_img[edge_nodes[0]] == 3)
        segment_list.append(edge_nodes)
    return segment_list


def pixel_to_coord(pixel):

    row, col = pixel
    min_x = -0.5
    min_y = -1532.5
    height = 1533
    cell_size = 2

    x = col * cell_size + min_x
    y = (height - row) * cell_size + min_y

    return (x, y)



class PixelToCoord():
    def __init__(self, min_x, min_y, height, cell_size):
        self.min_x = min_x
        self.min_y = min_y
        self.height = height
        self.cell_size = cell_size
    def __call__(self, pixel):
        row, col = pixel
        x = col * self.cell_size + self.min_x
        y = (self.height - row) * self.cell_size + self.min_y

        return (x, y)


def get_tif_transform(skeleton_file):
    with rasterio.open(skeleton_file) as src:

        height = src.height
        cell_size = src.transform[0]
        min_x = src.transform[2]
        max_y = src.transform[5]
        min_y = max_y - cell_size * height

        return min_x, min_y, height, cell_size, src.crs


def raster_to_shp(skeleton_file='./tif/skeleton_10.tif', shp_file_name='shp/line.shp'):


    min_x, min_y, height, cell_size, tif_crs = get_tif_transform(skeleton_file)
    pixel_to_coord = PixelToCoord(min_x, min_y, height, cell_size)

    print('find crossing')
    cross_img = find_crossing_point(skeleton_file)
    print('find main crossing')
    main_cross_img, crossing_main_dict = find_main_crossing(cross_img)
    print('find segment')
    segment_list = find_edges(main_cross_img)
    print('vectorize')

    nodes = {}
    intersections = []
    new_segment_list = []
    for segment in segment_list:
        
        new_segment = []
        head_node = crossing_main_dict[segment[0]].location

        if head_node not in nodes:
            nodes[head_node] = Node(*pixel_to_coord(head_node))
        
        new_segment.append(nodes[head_node])
        intersections.append(nodes[head_node])

        for i in range(1, len(segment) - 1):
            if segment[i] not in nodes:
                nodes[segment[i]] = Node(*pixel_to_coord(segment[i]))
            new_segment.append(nodes[segment[i]])

        tail_node = crossing_main_dict[segment[-1]].location

        if tail_node not in nodes:
            nodes[tail_node] = Node(*pixel_to_coord(tail_node))

        new_segment.append(nodes[tail_node])
        intersections.append(nodes[tail_node])

        new_segment_list.append(new_segment)
    
    lines = []
    for segment in new_segment_list:
        line = []
        
        for node in segment:
            line.append((node.x, node.y))
        line = LineString(line)
        lines.append(line)

        
    gdf = gpd.GeoDataFrame({
        'geometry': lines
    }, geometry='geometry', crs=tif_crs)

    gdf.to_file(shp_file_name)


if __name__ == '__main__':
    raster_to_shp(skeleton_file='./tif/skeleton.tif', shp_file_name='./result/line.shp')


