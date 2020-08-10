#!/usr/bin/python
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


class PreProcess(object):

    class Region:

        def __init__(self, left: int=0, top: int=0, right: int=0, bottom: int=0, debug: bool=False):
            if right < left:
                raise ValueError(f'Coordinate Error: right {right} is less than left {left}!')
            if bottom < top:
                raise ValueError(f'Coordinate Error: bottom {bottom} is less than top {top}!')
            self.debug  = debug
            self.left   = left
            self.top    = top
            self.right  = right
            self.bottom = bottom

        def get_height(self) -> int:
            return self.bottom - self.top
            
        def get_width(self) -> int:
            return self.right - self.left
            
        def shift_vert(self, displacement: int=0) -> bool:
            if (self.bottom + displacement) < 0 or \
               (self.top    + displacement) < 0:
                return False
            self.bottom += displacement
            self.top    += displacement
            return True
            
        def shift_hori(self, displacement: int=0) -> bool:
            if (self.left  + displacement) < 0 or \
               (self.right + displacement) < 0:
                return False
            self.left  += displacement
            self.right += displacement
            return True
            
        def is_empty(self) -> bool:
            return self.left   == 0 and \
                    self.right  == 0 and \
                   self.bottom == 0 and \
                   self.top    == 0
            
        def union(self, left: int, top: int, right: int, bottom: int) -> None:
            if self.debug:
                print(f'origin region: ({self.left}, {self.top}, {self.right}, {self.bottom})')
                print(f'new region: ({left}, {top}, {right}, {bottom})')
            if self.left > left:
                self.left = left
            if self.top > top:
                self.top = top
            if self.right < right:
                self.right = right
            if self.bottom < bottom:
                self.bottom = bottom
            if self.debug:
                print(f'merged region: ({self.left}, {self.top}, {self.right}, {self.bottom})')
        
        def contains(self, region) -> bool:
            return self.left   <= region.left  and \
                   self.right  >= region.right and \
                   self.top    <= region.top   and \
                   self.bottom >= region.bottom

    def __init__(self, input_file: str, debug: bool=False):
        if not os.path.isfile(input_file):
            raise Exception(f'Error: input file "{input_file}" is not found!')
        self.file = input_file
        self.debug = debug
        try:
            self.raw_image = cv2.imread(input_file)
            self.rgb_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
            self.gry_image = self.rgb_image[:,:,0]
        except:
            raise Exception(f'Error: loading image {input_file} failed!')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        if self.debug:
            plt.subplot(3,1,1)
            plt.imshow(self.raw_image)
            plt.subplot(3,1,2)
            plt.imshow(self.rgb_image)
            plt.subplot(3,1,3)
            plt.imshow(self.gry_image, cmap='gray')
            plt.show()
        height, width, _ = self.raw_image.shape
        self.raw_image_region = self.Region(0, 0, width, height, self.debug)
        self.raw_eye_region   = self.Region()
        self.locate_eye_region()
        self.crop_eye_region()

    def plot_subregion(self, sub_region) -> None:
        if not self.raw_eye_region.contains(sub_region):
            print(f'Warning: the region ({sub_region.left}, {sub_region.top}, ' +
                  f'{sub_region.right}, {sub_region.bottom}) is not completely ' +
                  f'inside the region ({self.raw_eye_region.left}, {self.raw_eye_region.top}, ' +
                  f'{self.raw_eye_region.right}, {self.raw_eye_region.bottom})!')
        rgb_image = self.rgb_image.copy()
        pt1       = (sub_region.left, sub_region.top)
        pt2       = (sub_region.right, sub_region.bottom)
        color     = (0, 255, 0)
        thickness = self.raw_eye_region.get_width() // 50
        cv2.rectangle(rgb_image, pt1, pt2, color, thickness)
        plt.imshow(rgb_image)
        plt.show()

    def locate_eye_region(self) -> bool:
        if self.gry_image is None:
            raise Exception('Error: image is None!')
        eyes = self.eye_cascade.detectMultiScale(self.gry_image)
        if len(eyes) == 0:
            print(f'Warning: no eye is detected in {self.file}!')
            return False
        for eye in eyes:
            x,y,w,h = eye
            if self.raw_eye_region.is_empty():
                self.raw_eye_region = self.Region(x, y, x+w, y+h, self.debug)
            else:
                self.raw_eye_region.union(x, y, x+w, y+h)
        if self.debug:
            self.plot_subregion(self.raw_eye_region)
        return True

    def crop_eye_region(self) -> None:
        dw = self.raw_eye_region.get_width() // 10
        left   = max(0, self.raw_eye_region.left - dw)
        right  = min(self.raw_eye_region.right + dw, self.raw_image_region.right)
        top    = self.raw_eye_region.top
        bottom = self.raw_eye_region.bottom
        self.rgb_image_cropped = self.rgb_image[top:bottom, left:right, :]
        if self.debug:
            plt.imshow(self.rgb_image_cropped)
            plt.show()

def test_main():
    parser = argparse.ArgumentParser(description='Pre-processing raw image')
    parser.add_argument('image_file', type=str, help='input image file')
    args = parser.parse_args()
    input_file = args.image_file
    prep = PreProcess(input_file, True)

if __name__ == '__main__':
    test_main()










