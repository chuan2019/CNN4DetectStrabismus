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
import numbers
from functools import total_ordering
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from detector import (
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEBUG,
    ShapeType,
    ImageType,
    logger,
    CLASSIFIER_EYE
)

class IntegerField:
    """
    IntegerField: data descriptor of Integral type
    """
    def __set_name__(self, owner_class, name):
        """set field name"""
        self.field_name = name # pylint: disable=W0201

    def __set__(self, instance, value):
        """data descriptor setter"""
        if not isinstance(value, numbers.Integral):
            raise TypeError(f'{self.field_name} must be integers')
        instance.__dict__[self.field_name] = value

    def __get__(self, instance, owner_class):
        """data descriptor getter"""
        if instance is None:
            return self
        return instance.__dict__.get(self.field_name, None)


class Vertex: # pylint: disable=R0903
    """
    Vertex: class of vertices on 2D plane
    """
    x = IntegerField()
    y = IntegerField() 

    def __init__(self, x: int=0, y: int=0): # pylint: disable=invalid-name
        '''
        the vertical axis in image coordinates is pointing downwards
        actual meaning of x and y coordinates are also different

        image coordinates:      normal coordinates:      ndarray indices:
         (0,0)  ----------> x       y ^                  [0, 0] -----------> y
                |                     |                         | [1, 1] [1, 2]
                |                     |                         | [2, 1] [2, 2]
                |                     |                         | [3, 1] [3, 2]
             y  v               (0,0) ----------> x           x v

        '''
        self.x = x # pylint: disable=C0103
        self.y = y # pylint: disable=C0103

    def __repr__(self):
        return f'({self.x}, {self.y})'


@total_ordering
class Region:
    """
        Region: class of rectanglular region on 2D plane
                all coordinates must be non-negative integers
    """

    def __init__(self, top_left: Vertex=Vertex(0,0), bottom_right: Vertex=Vertex(0,0)):
        '''
        constructing a Region object, default: empty region
        the vertical axis in image coordinates is pointing downwards

        image coordinates:      normal coordinates:      ndarray indices:
         (0,0)  ----------> x       y ^                  [0, 0] -----------> y
                |                     |                         | [1, 1] [1, 2]
                |                     |                         | [2, 1] [2, 2]
                |                     |                         | [3, 1] [3, 2]
             y  v               (0,0) ----------> x           x v


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
                f'top left corner: {top_left} is below the bottom right ' +
                f'corner: {bottom_right}'
            )
        if top_left.y > bottom_right.y:
            raise ValueError(
                f'bottom right corner: {bottom_right} is on the left of top ' +
                f'left corner: {top_left}!'
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

    @property
    def area(self) -> int:
        '''
        returns area of the region
        '''
        return self.width * self.height

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
        check image region containment
        :other: the other image region to be checked if inside the current region
        :return: True - the provided region is inside the current region;
                 False - the provided region is not inside the current region
        '''
        return self.left   <= other.left  and \
               self.right  >= other.right and \
               self.top    <= other.top   and \
               self.bottom >= other.bottom

    def __repr__(self):
        '''
        :return: the string representation of the region, which are the x and y
                 coordinates of its top-left corner and bottom-right corner
        '''
        return f'({self.left}, {self.top}, {self.right}, {self.bottom})'

    def __eq__(self, other: 'Region') -> bool:
        '''
        check if two regions have same area or not
        :other: the other image region to be checked if has same area as the current region
        :return: True - two regions are in same size;
                 False - two regions are in different sizes
        '''
        return self.area == other.area

    def __gt__(self, other: 'Region') -> bool:
        '''
        check if the provided image region is smaller than current image region
        :other: the other image region to be checked if smaller than the current region
        :return: True - the provided region is smaller than current region;
                 False - the provided region is not smaller than current region
        '''
        return self.area > other.area


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
        self._image     = None
        self._height    = 0
        self._width     = 0
        self._type      = None
        self._file      = ''
        self.img_region = Region()
        self.eye_region = Region()
        if src_img:
            if not os.path.isfile(src_img):
                raise IOError(f'{src_img} is not found!')
            logger.info('loading image from %s.', src_img)
            self.load(src_img)

    def set_image(self, _img: ImageType, _type: str) -> ShapeType:
        '''
        replace the current image and type with the provided image and type
        in this case, source image file becomes unknown (empty path string)
        :_img: provided (raw) image
        :_type: provided image (color) type, 'BGR', 'RGB', or 'GRAY'
        '''
        if not isinstance(_img, np.ndarray): # ImageType is just alias and used for type hint
            raise ValueError('ValueError: {type(_img)} is not of ImageType!')
        if _type not in ('BGR', 'RGB', 'GRAY'):
            raise ValueError('ValueError: image type {_type} is not supported!')
        self._type  = _type
        self._image = _img
        self._file  = ''
        self._height = _img.shape[0]
        self._width  = _img.shape[1]
        self.img_region = Region(Vertex(0, 0), Vertex(self._width, self._height))
        self.eye_region = Region()
        logger.warning('image is copied from other source, source file becomes UNKNOWN!')
        return _img.shape

    @property
    def type(self) -> str:
        '''
        return the type of the current image:
            RGB, BGR, GRAY
        '''
        return self._type

    @property
    def height(self) -> int:
        '''check height'''
        return self._height

    @property
    def width(self) -> int:
        '''check width'''
        return self._width

    def eye_located(self) -> bool:
        '''
        check if eye region is located from the image
        if eye region is not located, there could be two reasons:
            1. eye region is not searched yet
            2. there is no eye region in the image
        '''
        return not self.eye_region.is_empty()

    # pylint: disable=R0912
    def export(self, target_type: str) -> ImageType:
        '''
        export current image to RGB, BGR or GRAY image, current image remains intact
        '''
        if self._image is None:
            raise TypeError('cannot export empty image!')
        target_image = None
        if target_type == 'RGB':
            if self._type == 'RGB':
                target_image = self._image
            elif self._type == 'BGR':
                target_image = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
            else:
                raise TypeError(f'exporting {self._type} image to RGB type is not supported!')
        elif target_type == 'BGR':
            if self._type == 'BGR':
                target_image = self._image
            elif self._type == 'RGB':
                target_image = cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR)
            else:
                raise TypeError(
                    f'exporting {self._type} image to BGR type is not supported!'
                )
        elif target_type == 'GRAY':
            if self._type == 'GRAY':
                target_image = self._image
            elif self._type == 'BGR':
                target_image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
            elif self._type == 'RGB':
                target_image = cv2.cvtColor(self._image, cv2.COLOR_RGB2GRAY)
            else:
                raise TypeError(
                    f'converting {self._type} image to GRAYSCALE type is not supported!'
                )
        else:
            raise TypeError(
                f'exporting {self._type} image to {target_type} type is not supported!'
            )

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
        self._height, self._width, _ = self._image.shape # see the top Region comment
        self.img_region = Region(Vertex(), Vertex(self._width, self._height))
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

    def locate_eye_region(self) -> bool:
        '''
        search and locate eye region
        :return: True: eye region is found; False: eye region is not found
        '''
        if self._image is None:
            return False
        gry_image = None
        if self._type != 'GRAY':
            gry_image = self._image[:,:,0]
        else:
            gry_image = self._image
        if not os.path.isfile(CLASSIFIER_EYE):
            raise IOError(f'{CLASSIFIER_EYE} is not found!')
        eyes = cv2.CascadeClassifier(CLASSIFIER_EYE).detectMultiScale(gry_image)
        if len(eyes) < 2:
            logger.warning('%d eye is detected in %s!', len(eyes), self._file)
            return False
        eye_regions = []
        for eye in eyes:
            eye_x, eye_y, eye_w, eye_h = eye
            eye_region = Region(Vertex(eye_x, eye_y),
                                Vertex(eye_x + eye_w, eye_y + eye_h))
            # only keep the largest two eye regions
            ignored = False
            if len(eye_regions) == 0:
                eye_regions.append(eye_region)
            else:
                if eye_region > eye_regions[0]:
                    eye_regions.insert(0, eye_region)
                elif len(eye_regions) == 1:
                    eye_regions.insert(1, eye_region)
                elif eye_region > eye_regions[1]:
                    eye_regions[1] = eye_region
                else:
                    ignored = True
            if DEBUG:
                if ignored:
                    logger.info('Eye Region ignored: (%d, %d, %d, %d)',
                                 eye_x, eye_y, eye_x+eye_w, eye_y+eye_h)
                else:
                    logger.info('Eye Region detected: (%d, %d, %d, %d)',
                                 eye_x, eye_y, eye_x+eye_w, eye_y+eye_h)
                plt.imshow(gry_image, cmap='gray')
                # get current reference
                axes = plt.gca()
                # create a rectangle patch based on detected eye region
                if ignored:
                    rect = Rectangle((eye_x, eye_y), eye_w, eye_h,
                                     linewidth=2, edgecolor='r', facecolor='none')
                else:
                    rect = Rectangle((eye_x, eye_y), eye_w, eye_h,
                                     linewidth=2, edgecolor='g', facecolor='none')
                # add the patch to the Axes
                axes.add_patch(rect)
                plt.show()

        self.eye_region = eye_regions[0]
        self.eye_region.union(eye_regions[1])
        if DEBUG:
            logger.info('final eye region: (%d, %d, %d, %d)',
                        self.eye_region.left, self.eye_region.top,
                        self.eye_region.right, self.eye_region.bottom)
            plt.imshow(gry_image, cmap='gray')
            # get current reference
            axes = plt.gca()
            # create a rectangle patch based on detected eye region
            rect = Rectangle((self.eye_region.left, self.eye_region.top),
                             self.eye_region.width, self.eye_region.height,
                             linewidth=2, edgecolor='g', facecolor='none')
            # add the patch to the Axes
            axes.add_patch(rect)
            plt.show()

        return True

    def get_eye_region(self) -> Tuple[str, ImageType]:
        '''crop and return eye region, original image remains intact'''
        if self.eye_region.is_empty():
            logger.warning('eye region is empty, try locating it now ...')
            if not self.locate_eye_region():
                logger.error('no eye region is located!')
                return None
        #d_w    = self.eye_region.width // 10
        #left   = max(0, self.eye_region.left - d_w)
        #right  = min(self.eye_region.right + d_w, self.img_region.right)
        left   = self.eye_region.left
        right  = self.eye_region.right
        top    = self.eye_region.top
        bottom = self.eye_region.bottom
        eye_image = self._image[top:bottom, left:right, :] # see the top Region comment
        if DEBUG:
            logger.info('get_eye_region():\n********************************')
            logger.info('image region: (%d, %d, %d, %d)',
                        self.img_region.left, self.img_region.top,
                        self.img_region.right, self.img_region.bottom)
            logger.info('eye region: (%d, %d, %d, %d)',
                        self.eye_region.left, self.eye_region.top,
                        self.eye_region.right, self.eye_region.bottom)
            logger.info('crop region: (%d, %d, %d, %d)',
                        left, top, right, bottom)
            plt.imshow(eye_image[:,:,0], cmap='gray')
            plt.show()
        return (self._type, eye_image)
