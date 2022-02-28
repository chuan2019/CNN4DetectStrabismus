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
 File    : CNN4Strabismus.py

 Purpose : Implementation of CNN related features as listed below

        1. Raw Image Preprocessing
        2. Creating, Training, Saving and Loading a Selected CNN
           Model, for this feature, training/testing data sets
           are required
        3. Diagnosing a subject by classifying the input picture
           of the subject
"""
# pylint: skip-file
import argparse
from strabismus import logging
from strabismus.detector import StrabismusDetector

def main():
    '''
    sample command line:
    $ python CNN4Strabismus.py -m model_03-31-20 -i data/raw/patient/eso_008.jpg --raw
    $ python CNN4Strabismus.py -m model_03-31-20 -i data/raw/healthy/healthy_001.png --raw
    $ python CNN4Strabismus.py -m model_03-31-20 -i data/test/patient/1609_r.jpg
    $ python CNN4Strabismus.py -m model_03-31-20 -i data/test/healthy/806.jpg
    '''
    parser = argparse.ArgumentParser(description='Model Prediction')
    parser.add_argument('-m', '--model_file', type=str,
                        help='file name of the model to be loaded')
    parser.add_argument('-i', '--image_file', type=str,
                        help='file name of the image to be diagnosed')
    parser.add_argument('--raw', help='input image is not processed',
                        action='store_true')
    args = parser.parse_args()
    model_file = args.model_file
    image_file = args.image_file
    raw        = args.raw
    try:
        detector = StrabismusDetector(model_file, True)
        detector.is_strabismus(image_file, not raw)
    except Exception as err:
        print(str(err))
        logging.exception('%s', str(err))

if __name__ == '__main__':
    main()
