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
 File    : detector.py

 Purpose : classes for detecting strabismus from the preprocessed images

"""
import os
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
#from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from strabismus import (
    DEFAULT_WIDTH,
    DEFAULT_HIGHT,
    ModelType,
    ModelListType
)


class StrabismusDetector:
    """
    StrabismusDetector
    """

    class DetectorTraining:
        """
        DetectorTraining
        """

        def __init__(self,
                     training_set: str,
                     testing_set:  str,
                     model:        ModelType,
                     input_shape:  Tuple[int]=(DEFAULT_HIGHT, DEFAULT_WIDTH, 3),
                     batch_size:   int=16,
                     debug:        bool=False
                     ):
            if not os.path.isdir(training_set):
                raise Exception(f'Error: {training_set} is not a directory!')
            if not os.path.isdir(testing_set):
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
            except Exception as err:
                raise Exception(err) from err
            try:
                self.images_test  = self.get_image_generator(testing_set)
            except Exception as err:
                raise Exception(err) from err
            try:
                self._train_()
                self.model['trained'] = True
            except Exception as err:
                self.model['trained'] = False
                raise Exception(err) from err

        def get_image_generator(self,
                                directory: str,
                                rotation_range:int=10,
                                width_shift_range:float=0.0,
                                height_shift_range:float=0.0,
                                rescale:float=1/255,
                                horizontal_flip:bool=True,
                                fill_mode:str='nearest') -> object:
            '''get image generator'''
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
            '''train the model'''
            if self.images_train is None:
                raise Exception('Error: images for training are not loaded yet!')
            try:
                self.training_result = self.model['model'].fit_generator(self.images_train,
                                            epochs=self.epochs,
                                            steps_per_epoch=self.steps_per_epoch,
                                            validation_data=self.images_test,
                                            validation_steps=self.validation_steps)
            except Exception as err:
                raise Exception(err) from err

    class ModelFactory:
        """
        ModelFactory
        """

        def __init__(self, model_type:  str='LeNet',
                           model_name:  str=None,
                           input_shape: Tuple[int]=(DEFAULT_HIGHT, DEFAULT_WIDTH, 3)
                           #debug:       bool=False
                           ):
            if len(input_shape) != 3 or input_shape[2] != 3:
                raise Exception(f'Error: input shape, \"{input_shape}\", is not supported!')
            for num in input_shape:
                if num <= 0:
                    raise Exception(f'Error: entries of \"{input_shape}\" must positive integers!')
            self.input_shape = input_shape
            self.supported_model_types = ['LeNet', 'LeNet1']
            if model_name is None:
                model_name = model_type + '_' + str(dt.date.today().year)
                if dt.date.today().month < 10:
                    model_name += '-0' + str(dt.date.today().month)
                else:
                    model_name += '-' + str(dt.date.today().month)
                if dt.date.today().day < 10:
                    model_name += '-0' + str(dt.date.today().day)
                else:
                    model_name += '-' + str(dt.date.today().day)
            if model_type == 'LeNet':
                self.model = self.create_lenet(model_name)
            elif model_type == 'LeNet1':
                self.model = self.create_lenet1(model_name)
            else:
                raise Exception(f'Error: model type, \"{model_type}\", is not supported!')

        def get_model(self) -> ModelType:
            '''get model'''
            return self.model

        def create_lenet(self, name: str) -> ModelType:
            '''create LeNet'''
            lenet = Sequential()
            # 1st Convolution Layer
            lenet.add(Conv2D(filters=32,
                             kernel_size=(3,3),
                             input_shape=self.input_shape,
                             activation='relu'))
            # 1st Max Pooling (Subsampling) Layer
            lenet.add(MaxPooling2D(pool_size=(2,2)))
            # 2nd Convolution Layer
            lenet.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
            # 2nd Max Pooling (Subsampling) Layer
            lenet.add(MaxPooling2D(pool_size=(2,2)))
            # Flatten Layer
            lenet.add(Flatten())
            # 1st Full Connection Layer
            lenet.add(Dense(units=128, activation='relu'))
            # Output Layer
            lenet.add(Dense(units=1, activation='sigmoid'))
            lenet.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            return  {'name': name, 'trained': False, 'model': lenet}

        def create_lenet1(self, name: str) -> ModelType:
            '''create LeNet1'''
            lenet1 = Sequential()
            # 1st Convolution Layer
            lenet1.add(Conv2D(filters=32,
                              kernel_size=(3,3),
                              input_shape=self.input_shape,
                              activation='relu'))
            # 1st Pooling Layer
            lenet1.add(MaxPooling2D(pool_size=(2,2)))
            # 2nd Convolution Layer
            lenet1.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
            # 2nd Pooling Layer
            lenet1.add(MaxPooling2D(pool_size=(2,2)))
            # 3rd Convolution Layer
            lenet1.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
            # 3rd Pooling Layer
            lenet1.add(MaxPooling2D(pool_size=(2,2)))
            # Flatten Layer
            lenet1.add(Flatten())
            # 1st Full Connection Layer
            lenet1.add(Dense(units=128, activation='relu'))
            # Output Layer
            lenet1.add(Dense(units=1, activation='sigmoid'))
            lenet1.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
            return {'name': name, 'trained': False, 'model': lenet1}

    def __init__(self, model_name: str=None, debug: bool=False) -> None:
        self.debug = debug
        if model_name is None:
            self.model = None
        else:
            try:
                self._load_model_(model_name)
            except Exception as err:
                print(str(err))
                self.model = None

    def get_model_names(self) -> ModelListType:
        '''get model names'''
        model_files = []
        for (_, _, filename) in os.walk('models'):
            model_files.extend(filename)
        return model_files

    def _load_model_(self, model_name: str) -> bool:
        '''load model (private)'''
        model_file = 'models/' + model_name + '.h5'
        if self.debug:
            print(f'loading model file {model_file} ...')
        else:
            logging.info('loading model file %s ...', model_file)
        try:
            model = load_model(model_file)
            self.model = {'name': model_name, 'trained': True, 'model': model}
        except Exception as err:
            self.model = None
            if self.debug:
                print(str(err))
                print(f'Error: loading model \"{model_name}\" failed!')
            else:
                logging.error('%s', str(err))
                logging.error('Error: loading model \"%s\" failed!', model_name)
            return False
        if self.debug:
            print(f'model file {model_file} is loaded!')
        else:
            logging.info('model file %s is loaded!', model_file)
        return True

    def _save_model_(self, model_name: str=None, overwrite: bool=False) -> bool:
        '''save model (private)'''
        if self.model is None or self.model['trained'] is False:
            if self.debug:
                print('model is None or not trained!')
            else:
                logging.warning('model is None or not trained!')
            return False
        if model_name is None:
            model_name = self.model['name']
        model_file = 'models/' + model_name + '.h5'
        if not overwrite and os.path.isfile(model_file):
            if self.debug:
                print(f'model file {model_file} already exists!')
            else:
                logging.warning('model file %s already exists!', model_file)
            return False
        try:
            self.model['model'].save(model_file)
        except Exception as err:
            raise Exception(err) from err
        return True

    def create_model(self, model_name:str=None, model_type:str='LeNet') -> bool:
        '''create a CNN model'''
        try:
            self.model = self.ModelFactory(model_type=model_type,
                                           model_name=model_name
                                           #debug=self.debug
                                           ).get_model()
        except Exception as err:
            if self.debug:
                print(str(err))
                print(f'creating the {model_type}-type model {model_name} failed!')
            else:
                logging.error('%s', str(err))
                logging.error('creating the %s-type model %s failed!',
                              model_type, model_name)
            return False
        return True

    def train_model(self, training_set: str='../data/train',
                          testing_set: str='../data/test') -> bool:
        '''train the current CNN model'''
        if self.model is None:
            raise Exception('Error: model is not prepared yet,' +
                            'please either create or load one first!')
        try:
            self.DetectorTraining(training_set=training_set,
                                  testing_set=testing_set,
                                  model=self.model, debug=self.debug)
        except Exception as err:
            if self.debug:
                print(str(err))
                #print(f'creating and training the {model_type}-type model {model_name} failed!')
                # TODO: get model_type and model_name values properly
                print('creating and training the model failed!')
            else:
                logging.error('%s', str(err))
                #logging.error('creating and training the %s-type' +
                #              'model %s failed!', model_type, model_name)
                # TODO: get model_type and model_name values properly
                logging.error('creating and training the model failed!')
            return False
        return True

    def is_strabismus(self, input_image: str, processed: bool=False) -> bool:
        '''predict if the given subject has strabismus or not'''
        if self.model is None:
            raise Exception('Error: model is not loaded or created yet!')
        prep = PreProcess(debug=self.debug)
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
        else:
            logging.info('image.shape: (%d, %d)', image.shape[0], image.shape[1])
        try:
            predict = self.model['model'].predict_classes(image.reshape(dims))[[0]]
        except Exception as err:
            raise Exception(err) from err
        if self.debug:
            plt.figure(figsize=(5, 5))
            plt.imshow(image)
            if predict == 0:
                plt.title('Prediction: Healthy')
                print('the subject is diagnosed as healthy!')
            else:
                plt.title('Prediction: Strabismus')
                print('the subject is diagnosed as strabismus!')
            plt.show()
        return predict
