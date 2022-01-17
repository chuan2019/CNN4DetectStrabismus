"""
    Project: Detecting Strabismus with Convolutional Neural Networks

    Copyright (C) 2020  Chuan Zhang, chuan.zhang2015@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# --------------------------------------------------------------------------
 File    : preprocess.py

 Purpose : classes for preprocessing raw images

"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import NewType, Tuple, Union, List

ShapeType = NewType('ShapeType', Union[Tuple[int, int, int], Tuple[int, int]])

DEBUG = os.environ.get('CNN4DS_DEBUG', True)

class Vertex:
    """
    Vertex: class of vertices on 2D plane
    """

    def __init__(self, x: int=0, y: int=0):
        self._x = x
        self._y = y

    @property
    def x(self):
        '''check x value'''
        return self._x

    @property
    def y(self):
        '''check y value'''
        return self._y

    @x.setter
    def x(self, value):
        '''set x value'''
        self._x = value

    @y.setter
    def y(self, value):
        '''set y value'''
        self._y = value

    @x.deleter
    def x(self):
        '''delete variable _x'''
        del self._x

    @y.deleter
    def y(self):
        '''delete variable _y'''
        del self._y

    def __repr__(self):
        return f'({self._x}, {self._y})'


class Region:
    """
        Region: class of rectanglular region on 2D plane
                all coordinates must be non-negative integers
    """

    def __init__(self, v0: Vertex=Vertex(0,0), v1: Vertex=Vertex(0,0)):
        '''
        constructing a Region object, default: empty region
        the vertical axis in image coordinates is pointing downwards

        image coordinates:                 normal coordinates:
         (0,0)  ---------->                      ^
                |                                |
                |                                |
                |                                |
                v                          (0,0) ---------->

        :v0: top left corner
        :v1: bottom right corner
        '''
        try:
            self.setRegion(v0, v1)
        except ValueError as ve:
            raise ValueError(str(ve))

    def setRegion(self, v0: Vertex, v1: Vertex) -> None:
        '''
        :v0: top left corner
        :v1: bottom right corner
        '''
        if v0.x < 0 or v0.y < 0:
            raise ValueError(f'top left corner: {v0} is not in the first quadrant!')
        if v1.x < 0 or v1.y < 0:
            raise ValueError(f'bottom right corner: {v1} is not in the first quadrant!')
        if v0.x > v1.x:
            raise ValueError(f'bottom right corner: {v1} is on the left of top left corner: {v0}!')
        if v0.y > v1.y:
            raise ValueError(f'top left corner: {v0} is below the bottom right corner: {v1}')
        self.left   = v0.x
        self.top    = v0.y
        self.right  = v1.x
        self.bottom = v1.y


    @property
    def height(self) -> int:
        '''
        returns region height
        note: in image coordinates, y axis points downwards
        '''
        return self.bottom - self.top

    @property
    def width(self) -> int:
        '''
        returns region width
        '''
        return self.right - self.left

    def shift_vert(self, displacement: int=0):
        '''
        shift region vertically, positive direction is downward
        :displacement: positive - moving downward; negative - moving upward
        '''
        if (self.bottom + displacement) < 0 or \
            (self.top    + displacement) < 0:
            raise ValueError('Region shifted outside first quadrant!')
        self.bottom += displacement
        self.top    += displacement

    def shift_hori(self, displacement: int=0):
        '''
        shift region horizontally
        '''
        if (self.left  + displacement) < 0 or \
            (self.right + displacement) < 0:
            raise ValueError('Region shifted outside first quadrant!')
        self.left  += displacement
        self.right += displacement

    def is_empty(self) -> bool:
        '''
        check if the current image region is empty or not
        '''
        return self.left   == self.right and \
                self.bottom == self.top

    def union(self, other: 'Region') -> None:
        '''
        merge image regions by union
        :other: the other image region to be merged with current region
        '''
        if DEBUG:
            print(f'origin region: ({self.left}, {self.top}, {self.right}, {self.bottom})')
            print(f'new region: ({other.left}, {other.top}, {other.right}, {other.bottom})')
        self.left   = min(self.left,   other.left)
        self.top    = min(self.top,    other.top) # smaller y corresponds to upper location
        self.right  = max(self.right,  other.right)
        self.bottom = max(self.bottom, other.bottom) # larger y corresponds to lower location
        if DEBUG:
            print(f'merged region: ({self.left}, {self.top}, {self.right}, {self.bottom})')

    def intersect(self, other: 'Region') -> None:
        '''
        merge image regions by intersect
        :other: the other image region to be merged with current region
        '''
        if DEBUG:
            print(f'origin region: ({self.left}, {self.top}, {self.right}, {self.bottom})')
            print(f'new region: ({other.left}, {other.top}, {other.right}, {other.bottom})')
        self.left   = max(self.left,   other.left)
        self.top    = max(self.top,    other.top) # larger y corresponds to lower location
        self.right  = min(self.right,  other.right)
        self.bottom = min(self.bottom, other.bottom) # smaller y corresponds to upper location
        if DEBUG:
            print(f'merged region: ({self.left}, {self.top}, {self.right}, {self.bottom})')

    def contains(self, other: 'Region') -> bool:
        '''
        check image containment
        :other: the other image region to be checked if inside the current region
        '''
        return self.left   <= other.left  and \
               self.right  >= other.right and \
               self.top    <= other.top   and \
               self.bottom >= other.bottom

    def __repr__(self):
        return f'({self.left}, {self.top}, {self.right}, {self.bottom})'

class Image:
    """
    class of images and basic operations
    """

    def __init__(self):
        self._image  = None
        self._height = 0
        self._width  = 0
        self._type   = None
        self._file   = ''

    @property
    def type(self):
        '''
        return the type of the current image:
            RGB, BGR, GRAY
        '''
        return self._type

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def toRGB(self) -> ShapeType:
        '''
        export current image to RGB image, current image remains intact
        '''
        if self._image is None:
            raise TypeError('cannot convert empty image!')
        target_image = None
        if self._type == 'RGB':
            target_image = self._image
        elif self._type == 'BGR':
            target_image = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        else:
            raise TypeError(f'converting {self._type} image to RGB type is not supported!')
        return target_image

    def toBGR(self) -> ShapeType:
        '''
        export current image to BGR image, current image remains intact
        '''
        if self._image is None:
            raise TypeError('cannot convert empty image!')
        target_image = None
        if self._type == 'BGR':
            target_image = self._image
        elif self._type == 'RGB':
            target_image = cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError(f'converting {self._type} image to BGR type is not supported!')

    def toGRAY(self) -> ShapeType:
        '''
        export current image to GRAY image, current image remains intact
        '''
        if self._image is None:
            raise TypeError('cannot convert empty image!')
        target_image = None
        if self._type == 'GRAY':
            target_image = self._image
        elif self._type == 'BGR' or self._type == 'RGB':
            target_image = Image()
            target_image

    def load(self, input_file: str) -> ShapeType:
        '''load a BGR image from file'''
        if not os.path.isfile(input_file):
            raise IOError(f'image file "{input_file}" is not found!')
        self._file = input_file
        try:
            self._image = cv2.imread(input_file)
            self._type  = 'BGR'
        except Exception as err:
            raise Exception(err) from err

    def show(self) -> None:
        if self._type == 'BGR':
            image = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        elif self._type == 'RGB':
            image = self._image
        elif self._type == 'GRAY':
            image = cv2.cvtColor(self._image, cv2.COLOR_GRAY2RGB)
        else:
            raise TypeError('only BGR, RGB and GRAYSCALE images are supported')
        plt.imshow(image)
        plt.show()

    def resize(self, width: int=400, height: int=100) -> None:
        '''resize image'''
        dim = (width, height)
        self.rgb_image_cropped = cv2.resize(self.rgb_image_cropped,
                                            dim,
                                            interpolation=cv2.INTER_AREA)
        self.gry_image_cropped = self.rgb_image_cropped[:, :, 0]
        if DEBUG:
            img_h, img_w, _ = self.rgb_image_cropped.shape
            print(f'dim after resize: {(img_h, img_w)}')

