import os
import sys
import logging
import datetime as dt
from typing import NewType, Tuple, Union, List

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

HEADER_TEXT  = '\n\tCNN4Strabismus\n\n'
HEADER_TEXT += 'Copyright (C) 2020-2022 Chuan Zhang\n\n\n'
#print(HEADER_TEXT)

LOGFILE_NAME = '/var/log/cnn4strabismus/model_' + '-'.join([str(dt.date.today().year),
                                    str(dt.date.today().month),
                                    str(dt.date.today().day)]) + '.log'

logging.basicConfig(filename=LOGFILE_NAME,
                    #filemode='w',
                    format='%(asctime)s  [%(levelname)s:%(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

# default shape:
#   + target shape for preprocessing
#   + input  shape for detecting
DEFAULT_HIGHT = 100
DEFAULT_WIDTH = 400

DEBUG = os.environ.get('CNN4DS_DEBUG', True)

ShapeType = NewType('ShapeType', Union[Tuple[int, int, int], Tuple[int, int]])
