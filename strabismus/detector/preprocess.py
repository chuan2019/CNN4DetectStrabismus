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
from typing import Tuple
import os
import cv2
import matplotlib.pyplot as plt
from detector import (
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEBUG,
    ShapeType,
    ImageType,
    logger
)

class Vertex:
    """
    Vertex: class of vertices on 2D plane
    """

    def __init__(self, x: int=0, y: int=0):
        self._x = x
        self._y = y

    @property
    def x(self): # pylint: disable=invalid-name
        '''check x value'''
        return self._x

    @property
    def y(self): # pylint: disable=invalid-name
        '''check y value'''
        return self._y

    @x.setter
    def x(self, value): # pylint: disable=invalid-name
        '''set x value'''
        self._x = value

    @y.setter
    def y(self, value): # pylint: disable=invalid-name
        '''set y value'''
        self._y = value

    @x.deleter
    def x(self): # pylint: disable=invalid-name
        '''delete variable _x'''
        del self._x

    @y.deleter
    def y(self): # pylint: disable=invalid-name
        '''delete variable _y'''
        del self._y

    def __repr__(self):
        return f'({self._x}, {self._y})'


class Region:
    """
        Region: class of rectanglular region on 2D plane
                all coordinates must be non-negative integers
    """

    def __init__(self, top_left: Vertex=Vertex(0,0), bottom_right: Vertex=Vertex(0,0)):
        '''
        constructing a Region object, default: empty region
        the vertical axis in image coordinates is pointing downwards

        image coordinates:                 normal coordinates:
         (0,0)  ---------->                      ^
                |                                |
                |                                |
                |                                |
                v                          (0,0) ---------->

        :top_left:     top left corner
        :bottom_right: bottom right corner
        '''
        self.top    = 0
        self.left   = 0
        self.bottom = 0
        self.right  = 0
        try:
            self.set_region(top_left, bottom_right)
        except ValueError as err:
            raise ValueError(str(err)) from err

    def set_region(self, top_left: Vertex, bottom_right: Vertex) -> None:
        '''
        :top_left: top left corner
        :bottom_right: bottom right corner
        '''
        if top_left.x < 0 or top_left.y < 0:
            raise ValueError(
                f'top left corner: {top_left} is not in the first quadrant!'
            )
        if bottom_right.x < 0 or bottom_right.y < 0:
            raise ValueError(
                f'bottom right corner: {bottom_right} is not in the first quadrant!'
            )
        if top_left.x > bottom_right.x:
            raise ValueError(
                f'bottom right corner: {bottom_right} is on the left of top ' +
                f'left corner: {top_left}!'
            )
        if top_left.y > bottom_right.y:
            raise ValueError(
                f'top left corner: {top_left} is below the bottom right ' +
                f'corner: {bottom_right}'
            )
        self.left   = top_left.x
        self.top    = top_left.y
        self.right  = bottom_right.x
        self.bottom = bottom_right.y


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
            logger.debug('origin region: (%d, %d, %d, %d)',
                         self.left, self.top, self.right, self.bottom)
            logger.debug('new region: (%d, %d, %d, %d)',
                         other.left, other.top, other.right, other.bottom)
        self.left   = min(self.left,   other.left)
        self.top    = min(self.top,    other.top) # smaller y corresponds to upper location
        self.right  = max(self.right,  other.right)
        self.bottom = max(self.bottom, other.bottom) # larger y corresponds to lower location
        if DEBUG:
            logger.debug('merged region: (%d, %d, %d, %d)',
                         self.left, self.top, self.right, self.bottom)

    def intersect(self, other: 'Region') -> None:
        '''
        merge image regions by intersect
        :other: the other image region to be merged with current region
        '''
        if DEBUG:
            logger.debug('origin region: (%d, %d, %d, %d)',
                         self.left, self.top, self.right, self.bottom)
            logger.debug('new region: (%d, %d, %d, %d)',
                         other.left, other.top, other.right, other.bottom)
        self.left   = max(self.left,   other.left)
        self.top    = max(self.top,    other.top) # larger y corresponds to lower location
        self.right  = min(self.right,  other.right)
        self.bottom = min(self.bottom, other.bottom) # smaller y corresponds to upper location
        if DEBUG:
            logger.debug('merged region: (%d, %d, %d, %d)',
                         self.left, self.top, self.right, self.bottom)

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

    def __init__(self, src_img: str=''):
        '''
        constructing Image object in two ways:
          1. default: empty Image object
          2. load image from the provided image path: src_img
        '''
        self._image  = None
        self._height = 0
        self._width  = 0
        self._type   = None
        self._file   = ''
        if os.path.isfile(src_img):
            self.load(src_img)

    def set_image(self, _img: ImageType, _type: str) -> None:
        '''
        replace the current image and type with the provided image and type
        in this case, source image file becomes unknown (empty path string)
        :_img: provided (raw) image
        :_type: provided image (color) type, 'BGR', 'RGB', or 'GRAY'
        '''
        if not isinstance(_img, ImageType):
            raise ValueError('ValueError: {type(_img)} is not of ImageType!')
        if _type not in ('BGR', 'RGB', 'GRAY'):
            raise ValueError('ValueError: image type {_type} is not supported!')
        self._type  = _type
        self._image = _img
        self._file  = ''

    @property
    def type(self):
        '''
        return the type of the current image:
            RGB, BGR, GRAY
        '''
        return self._type

    @property
    def height(self):
        '''check height'''
        return self._height

    @property
    def width(self):
        '''check width'''
        return self._width

    def to_rgb(self) -> ImageType:
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

    def to_bgr(self) -> ImageType:
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
        return target_image

    def to_gray(self) -> ImageType:
        '''
        export current image to GRAY image, current image remains intact
        '''
        if self._image is None:
            raise TypeError('cannot convert empty image!')
        target_image = None
        if self._type == 'GRAY':
            target_image = self._image
        elif self._type == 'BGR':
            target_image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        elif self._type == 'RGB':
            target_image = cv2.cvtColor(self._image, cv2.COLOR_RGB2GRAY)
        else:
            raise TypeError(f'converting {self._type} image to GRAYSCALE type is not supported!')
        return target_image

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
        self._height, self._width, _ = self._image.shape
        if DEBUG:
            logger.debug('image shape: (%d, %d, %d)',
                         self._image.shape[0], self._image.shape[1], self._image.shape[2])

    def show(self) -> None:
        '''converting current image to RGB type, then display'''
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

    def resize(self, width: int=DEFAULT_WIDTH, height: int=DEFAULT_HEIGHT) -> Tuple[str, ImageType]:
        '''
        resize image: rescale the current image to the target size
        '''
        if width <= 0 or height <= 0:
            raise ValueError(f'ValueError: width={width}, height={height}!')
        dim = (width, height)
        img = cv2.resize(self._image, dim, interpolation=cv2.INTER_AREA)
        if DEBUG:
            img_h, img_w, _ = img.shape
            logger.debug('dim after resize: (%d, %d)', img_h, img_w)
        return (self.type, img)
