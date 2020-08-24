#!/usr/bin/python
#######################################################################
# Project: Detecting Strabismus with Convolutional Neural Networks
# Author : Chuan Zhang
# Email  : chuan.zhang2015@gmail.com
# Date   : Feb. 2020 - Aug. 2020
#######################################################################
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import logging
import datetime as dt
from typing import NewType, Tuple, Union, List
from abc import ABC, abstractmethod
import asyncio

logfile_name = 'model_' + '-'.join([str(dt.date.today().year),
                                    str(dt.date.today().month),
                                    str(dt.date.today().day)]) + '.log'

logging.basicConfig(filename=logfile_name,
                    filemode='w',
                    format='%(asctime)s  [%(levelname)s:%(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

ModelType = NewType('ModelType', {'name':str, 'trained': bool, 'model': Sequential})
ImageType = NewType('ImageType', np.ndarray)
ShapeType = NewType('ShapeType', Union[Tuple[int, int, int], Tuple[int, int]])
ModelListType = NewType('ModelListType', List[str])

class PreProcess(object):

    class Region:

        def __init__(self, left: int=0, top: int=0, right: int=0, bottom: int=0, debug: bool=False):
            if right < left:
                logging.error(f'Coordinate Error: right {right} is less than left {left}!')
                raise ValueError(f'Coordinate Error: right {right} is less than left {left}!')
            if bottom < top:
                logging.error(f'Coordinate Error: bottom {bottom} is less than top {top}!')
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
                logging.debug(f'origin region: ({self.left}, {self.top}, {self.right}, {self.bottom})')
                logging.debug(f'new region: ({left}, {top}, {right}, {bottom})')
            if self.left > left:
                self.left = left
            if self.top > top:
                self.top = top
            if self.right < right:
                self.right = right
            if self.bottom < bottom:
                self.bottom = bottom
            if self.debug:
                logging.debug(f'merged region: ({self.left}, {self.top}, {self.right}, {self.bottom})')
        
        def contains(self, region) -> bool:
            return self.left   <= region.left  and \
                   self.right  >= region.right and \
                   self.top    <= region.top   and \
                   self.bottom >= region.bottom

    def __init__(self, debug: bool=False):
        self.debug = debug
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.raw_image = None
        self.rgb_image = None
        self.gry_image = None
        self.raw_eye_region    = None
        self.raw_image_region  = None
        self.rgb_image_cropped = None
        self.gry_image_cropped = None
        self.WIDTH  = 350
        self.HEIGHT = 100

    def load_image(self, input_file: str) -> ShapeType:
        if not os.path.isfile(input_file):
            logging.error(f'input file "{input_file}" is not found!')
            raise Exception(f'Error: input file "{input_file}" is not found!')
        self.file = input_file
        try:
            self.raw_image = cv2.imread(input_file)
            self.rgb_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
            self.gry_image = self.rgb_image[:,:,0]
        except:
            logging.error(f'loading image {input_file} failed!')
            raise Exception(f'Error: loading image {input_file} failed!')
        if self.debug:
            plt.subplot(3,1,1)
            plt.imshow(self.raw_image)
            plt.subplot(3,1,2)
            plt.imshow(self.rgb_image)
            plt.subplot(3,1,3)
            plt.imshow(self.gry_image, cmap='gray')
            plt.show()
        return self.raw_image.shape

    def set_image(self, input_image: ImageType, image_type: str='bgr') -> ShapeType:
        '''
        Get pre-loaded image, assume the image is in BGR format
        '''
        if input_image is None:
            raise Exception(f'Error: input image is None!')
        if len(input_image.shape) != 3:
            raise Exception(f'Error: input image shape is not supported!')
        if image_type.upper() == 'BGR':
            self.raw_image = input_image
            self.rgb_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
        elif image_type.upper() == 'RGB':
            self.rgb_image = input_image
            self.raw_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
        else:
            raise Exception(f'Error: image type {image_type} is not supported!')
        self.gry_image = self.rgb_image[:,:,0]
        self.rgb_image_cropped = self.rgb_image
        self.gry_image_cropped = self.gry_image
        if self.debug:
            plt.subplot(3,1,1)
            plt.imshow(self.raw_image)
            plt.subplot(3,1,2)
            plt.imshow(self.rgb_image)
            plt.subplot(3,1,3)
            plt.imshow(self.gry_image, cmap='gray')
            plt.show()
        return self.raw_image.shape

    def preprocess(self) -> None:
        '''
        locate and extract eye region
        '''
        height, width, _ = self.raw_image.shape
        self.raw_image_region = self.Region(0, 0, width, height, self.debug)
        self.raw_eye_region   = self.Region()
        self.locate_eye_region()
        self.crop_eye_region()
    
    def get_processed_image(self, image_type: str) -> ImageType:
        '''
        get extracted eye region either in RGB or GRAY
        '''
        height, width, _ = self.rgb_image_cropped.shape
        if self.debug:
            print(f'dim: {(height, width)}')
        if width != self.WIDTH or height != self.HEIGHT:
            self.resize_image(self.WIDTH, self.HEIGHT)
        if image_type.upper() == 'RGB':
            return self.rgb_image_cropped
        elif image_type.upper() == 'GRAY':
            return self.gry_image_cropped

    def plot_subregion(self, sub_region) -> None:
        if not self.raw_eye_region.contains(sub_region):
            logging.warning(f'the region ({sub_region.left}, {sub_region.top}, ' +
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
            logging.error('image is None!')
            raise Exception('Error: image is None!')
        eyes = self.eye_cascade.detectMultiScale(self.gry_image)
        if len(eyes) == 0:
            logging.warning(f'no eye is detected in {self.file}!')
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
        self.gry_image_cropped = self.rgb_image_cropped[:, :, 0]
    
    def resize_image(self, width: int=400, height: int=100) -> None:
        dim = (width, height)
        self.rgb_image_cropped = cv2.resize(self.rgb_image_cropped, dim, interpolation=cv2.INTER_AREA)
        self.gry_image_cropped = self.rgb_image_cropped[:, :, 0]
        if self.debug:
            h, w, c = self.rgb_image_cropped.shape
            print(f'dim after resize: {(h, w)}')

def test_preprocessing():
    parser = argparse.ArgumentParser(description='Pre-processing raw image')
    parser.add_argument('-f', '--image_file', type=str, help='input image file')
    args = parser.parse_args()
    input_file = args.image_file
    prep = PreProcess(True)
    shape = prep.load_image(input_file)
    print(f'loaded image from file {input_file}, image shape: {shape}.')
    prep.preprocess()
    rgb_cropped = prep.get_processed_image('rgb')
    gry_cropped = prep.get_processed_image('gray')
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(rgb_cropped)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(gry_cropped, cmap='gray')
    plt.show()

class StrabismusDetector(object):

    class DetectorTraining(object):

        def __init__(self, 
                     training_set: str,
                     testing_set:  str,
                     model:        ModelType,
                     input_shape:  tuple=(100, 400, 3),
                     batch_size:   int=16,
                     debug:        bool=False
                     ):
            if not os.path.isdir(training_set):
                #logging.critical(f'{training_set} is not a directory!')
                raise Exception(f'Error: {training_set} is not a directory!')
            if not os.path.isdir(testing_set):
                #logging.critical(f'{testing_set} is not a directory!')
                raise Exception(f'Error: {testing_set} is not a directory!')
            self.training_set     = training_set
            self.testing_set      = testing_set
            self.model            = model
            self.model['trained'] = False
            self.input_shape      = input_shape
            self.batch_size       = batch_size
            self.images_train     = None
            self.images_test      = None
            self.debug            = debug
            if debug:
                self.epochs           = 2
                self.steps_per_epoch  = 5
                self.validation_steps = 3
            else:
                self.epochs           = 50
                self.steps_per_epoch  = 150
                self.validation_steps = 12
            try:
                self.images_train = self.get_image_generator(training_set)
            except:
                #logging.critical(f'preparing data set from {training_set} for training failed!')
                raise Exception(f'Error: preparing data set from {training_set} for training failed!')
            try:
                self.images_test  = self.get_image_generator(testing_set)
            except:
                #logging.critical(f'preparing data set from {testing_set} for testing failed!')
                raise Exception(f'Error: preparing data set from {testing_set} for testing failed!')
            try:
                self._train_()
                self.model['trained'] = True
            except:
                self.model['trained'] = False

        def get_image_generator(self,
                                directory: str,
                                rotation_range:int=10,
                                width_shift_range:float=0.0,
                                height_shift_range:float=0.0,
                                rescale:float=1/255,
                                horizontal_flip:bool=True,
                                fill_mode:str='nearest') -> object:
            image_gen = ImageDataGenerator(rotation_range,
                                           width_shift_range,
                                           height_shift_range,
                                           rescale,
                                           horizontal_flip,
                                           fill_mode)
            images = image_gen.flow_from_directory(directory=directory,
                                                   target_size=self.input_shape[:2],
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   class_mode='binary')
            return images

        def _train_(self) -> None:
            if self.images_train is None:
                #logging.error('images for training are not loaded yet!')
                raise Exception('Error: images for training are not loaded yet!')
            try:
                self.training_result = self.model['model'].fit_generator(self.images_train,
                                            epochs=self.epochs,
                                            steps_per_epoch=self.steps_per_epoch,
                                            validation_data=self.images_test,
                                            validation_steps=self.validation_steps)
            except:
                #logging.error('training model failed!')
                raise Exception('Error: training model failed!')

    class ModelFactory(object):

        def __init__(self, model_name: str, model_type: str='LeNet', debug: bool=False):
            # Supported model names: ['LeNet', 'LeNet1']
            if model_type == 'LeNet':
                self.model = self.create_LeNet(model_name)
            elif model_type == 'LeNet1':
                self.model = self.create_LeNet1(model_name)
            else:
                raise Exception(f'Error: {model_type} is not a supported model type!')

        def get_model(self) -> ModelType:
            return self.model

        def create_LeNet(self, name: str) -> ModelType:
            LeNet = Sequential()
            # 1st Convolution Layer
            LeNet.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=input_size, activation='relu'))
            # 1st Max Pooling (Subsampling) Layer
            LeNet.add(MaxPooling2D(pool_size=(2,2)))
            # 2nd Convolution Layer
            LeNet.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
            # 2nd Max Pooling (Subsampling) Layer
            LeNet.add(MaxPooling2D(pool_size=(2,2)))
            # Flatten Layer
            LeNet.add(Flatten())
            # 1st Full Connection Layer
            LeNet.add(Dense(units=128, activation='relu'))
            # Output Layer
            LeNet.add(Dense(units=1, activation='sigmoid'))
            LeNet.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            return  {'name': name, 'trained': False, 'model': LeNet}

        def create_LeNet1(self, name: str) -> ModelType:
            LeNet1 = Sequential()
            # 1st Convolution Layer
            LeNet1.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=input_size, activation='relu'))
            # 1st Pooling Layer
            LeNet1.add(MaxPooling2D(pool_size=(2,2)))
            # 2nd Convolution Layer
            LeNet1.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
            # 2nd Pooling Layer
            LeNet1.add(MaxPooling2D(pool_size=(2,2)))
            # 3rd Convolution Layer
            LeNet1.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
            # 3rd Pooling Layer
            LeNet1.add(MaxPooling2D(pool_size=(2,2)))
            # Flatten Layer
            LeNet1.add(Flatten())
            # 1st Full Connection Layer
            LeNet1.add(Dense(units=128, activation='relu'))
            # Output Layer
            LeNet1.add(Dense(units=1, activation='sigmoid'))
            LeNet1.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            return {'name': name, 'trained': False, 'model': LeNet1}

    def __init__(self, model_file: str=None, debug: bool=False) -> None:
        if model_file is None:
            self.model = None
        else:
            try:
                self._load_model_(model_file)
            except:
                self.model = None
        self.debug   = debug

    def get_model_names(self) -> ModelListType:
        model_files = []
        for (dirpath, dirname, filename) in os.walk('models'):
            model_files.extend(filename)
        return model_files

    def _load_model_(self, model_file: str) -> bool:
        try:
            if 'models/' not in model_file:
                model_file = 'models/' + model_file
            model = load_model(model_file)
            self.model = {'name': model_file, 'trained': True, 'model': model}
        except:
            if os.path.isfile(model_file):
                #logging.error(f'model file {model_file} is not found!')
                raise Exception(f'Error: model file {model_file} is not found!')
            else:
                #logging.error(f'loading model {model_file} failed!')
                raise Exception(f'Error: loading model {model_file} failed!')
        return True

    def _save_model_(self, model_file: str) -> bool:
        if self.model is None or self.model['trained'] == False:
            logging.warning(f'model is None or not trained!')
            return False
        if 'models/' not in model_file:
            model_file = 'models/' + model_file
        if os.path.isfile(model_file):
            logging.warning(f'model {model_file} already exists!')
            return False
        self.model['model'].save(model_file)
        return True

    def create_model(self, model_name:str, model_type:str='LeNet') -> bool:
        '''
        create and train a CNN model
        '''
        try:
            self.model = self.ModelFactory(model_name, model_type, self.debug).get_model()
            self.DetectorTraining(training_set='../data/train', testing_set='../data/test',
                                  model=self.model, debug=self.debug)
        except:
            print(f'creating and training the {model_type}-type model {model_name} failed!')
            return False
        return True

    def isStrabismus(self, input_image: str, processed: bool=False) -> bool:
        if self.model is None:
            raise Exception(f'Error: model is not loaded or created yet!')
        prep = PreProcess(self.debug)
        if not processed:
            prep.load_image(input_image)
            prep.preprocess()
        else:
            prep.set_image(cv2.imread(input_image))
        image = prep.get_processed_image('rgb')
    
        dims = [1]
        dims.extend(list(image.shape))
        if self.debug:
            print(f'image.shape: {image.shape}')
        try:
            predict = self.model['model'].predict_classes(image.reshape(dims))[[0]]
        except:
            raise Exception(f'Error: diagnosis failed! Input image may not be supported!')
        if self.debug:
            fig = plt.figure(figsize=(5, 5))
            plt.imshow(image)
            if predict == 0:
                plt.title('Prediction: Healthy')
                print('the subject is diagnosed as healthy!')
            else:
                plt.title('Prediction: Strabismus')
                print('the subject is diagnosed as strabismus!')
            plt.show()
        return predict

def test_model_training() -> None:
    ...

def test_model_prediction() -> None:
    parser = argparse.ArgumentParser(description='Test Model Prediction')
    parser.add_argument('-m', '--model_file', type=str, help='file name of the model to be loaded')
    parser.add_argument('-i', '--image_file', type=str, help='file name of the image to be diagnosed')
    parser.add_argument('--raw', help='input image is not processed', action='store_true')
    args = parser.parse_args()
    model_file = args.model_file
    image_file = args.image_file
    raw        = args.raw
    detector = StrabismusDetector(model_file, True)
    detector.isStrabismus(image_file, not raw)

if __name__ == '__main__':
    #test_preprocessing()
    test_model_prediction()










