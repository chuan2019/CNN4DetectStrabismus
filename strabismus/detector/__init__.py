import os
import sys
import logging
import datetime as dt

#############################
#   package path            #
#############################
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

#############################
# Logging Configurations    #
#############################
LOGFILE_NAME = '../logs/model_' + '-'.join([str(dt.date.today().year),
                                    str(dt.date.today().month),
                                    str(dt.date.today().day)]) + '.log'

logging.basicConfig(filename=LOGFILE_NAME,
                    #filemode='w',
                    format='%(asctime)s  [%(levelname)s:%(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

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


