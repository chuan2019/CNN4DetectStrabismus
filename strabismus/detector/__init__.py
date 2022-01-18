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
"""
import os
import sys
import logging
import datetime as dt
from typing import NewType, Tuple, Union, List
import numpy as np
from keras.models import Sequential

#############################
#   package path            #
#############################
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

#############################
# Logging Configurations    #
#############################


log_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../logs/'
)
log_file_name = 'model_' + '-'.join([str(dt.date.today().year),
                                    str(dt.date.today().month),
                                    str(dt.date.today().day)]) + '.log'

LOGFILE_NAME = log_dir + '/' + log_file_name

logging.basicConfig(filename=LOGFILE_NAME,
                    #filemode='w',
                    format='%(asctime)s  [%(levelname)s:%(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger('strabismus.detector.logger')

#############################
# global variables/type     #
#############################
HEADER_TEXT  = '\n\tCNN4Strabismus\n\n'
HEADER_TEXT += 'Copyright (C) 2020-2022 Chuan Zhang\n\n\n'
#print(HEADER_TEXT)

# default shape:
#   + target shape for preprocessing
#   + input  shape for detecting
DEFAULT_HEIGHT = 100
DEFAULT_WIDTH = 400

DEBUG = os.environ.get('CNN4DS_DEBUG', True)

ShapeType = NewType('ShapeType', Union[Tuple[int, int, int], Tuple[int, int]])
ImageType = NewType('ImageType', np.ndarray)

ModelType = NewType('ModelType', {'name': str, 'trained': bool, 'model': Sequential})
ModelListType = NewType('ModelListType', List[str])

CLASSIFIER_EYE = '../classifiers/haarcascade_eye.xml'
