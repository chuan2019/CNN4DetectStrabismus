#!/usr/bin/python
#######################################################################
# Project: Detecting Strabismus with Convolutional Neural Networks
# File   : CNN4Strabismus.py
# Purpose: Reorganize and Update unit test cases
# Author : Chuan Zhang
# Email  : chuan.zhang2015@gmail.com
# Example: $ python -m unittest -v CNN4StrabismusUTs.py
#######################################################################

from ../cnn_4_strabismus import *
import unittest

class PreprocessingTestSuite(unittest.TestCase):

    def setUp(self):
        self.raw_image_healthy = [
            {'file_name':'data/raw/healthy/healthy_001.png',
             'shape':(1138, 2086, 3)},
            {'file_name':'data/raw/healthy/healthy_014.jpg',
             'shape':(640, 428, 3)},
            {'file_name':'data/raw/healthy/healthy_016.jpg',
             'shape':(800, 554, 3)},
            {'file_name':'data/raw/healthy/healthy_017.jpg',
             'shape':(260, 173, 3)}
        ]
        self.raw_image_strabismus = [
            {'file_name': 'data/raw/patient/eso_008.jpg',
             'shape': (467, 600, 3)},
            {'file_name':'data/raw/patient/exo_004.jpg',
             'shape':(150, 200, 3)},
            {'file_name':'data/raw/patient/hyper_002.jpg',
             'shape':(797, 1063, 3)},
            {'file_name':'data/raw/patient/hypo_002.jpg',
             'shape':(1107, 800, 3)}
        ]
        self.prep = PreProcess(debug=False)

    def testLoadImage(self):
        failed = False
        try:
            for idx in range(len(self.raw_image_healthy)):
                shape = self.prep.load_image(self.raw_image_healthy[idx]['file_name'])
                self.assertEqual(self.raw_image_healthy[idx]['shape'], shape)
            for idx in range(len(self.raw_image_strabismus)):
                shape = self.prep.load_image(self.raw_image_strabismus[1]['file_name'])
                self.assertEqual(self.raw_image_strabismus[1]['shape'], shape)
        except Exception as err:
            print(str(err))
            failed = True
        self.assertFalse(failed)

    def testCroppingImage(self):
        failed = False
        try:
            self.prep.load_image(self.raw_image_healthy[0]['file_name'])
            self.prep.preprocess()
            rgb_cropped = self.prep.get_processed_image('rgb')
            self.assertEqual(type(rgb_cropped).__name__, 'ndarray')
            self.assertEqual(rgb_cropped.shape, (self.prep.HEIGHT, self.prep.WIDTH, 3))
            gry_cropped = self.prep.get_processed_image('gray')
            self.assertEqual(type(gry_cropped).__name__, 'ndarray')
            self.assertEqual(gry_cropped.shape, (self.prep.HEIGHT, self.prep.WIDTH))
        except Exception as err:
            print(str(err))
            failed = True
        self.assertFalse(failed)


class ModelPredictionTestSuite(unittest.TestCase):

    def setUp(self):
        self.models = [
            'chuan_cnn',
            'chuan_cnn2',
            'model_03-31-20'
        ]
        self.raw_image_healthy = [
            'data/raw/healthy/healthy_001.png',
            'data/raw/healthy/healthy_014.jpg',
            'data/raw/healthy/healthy_016.jpg',
            'data/raw/healthy/healthy_017.jpg'
        ]
        self.raw_image_patient = [
            'data/raw/patient/eso_008.jpg',
            'data/raw/patient/exo_004.jpg',
            'data/raw/patient/hyper_002.jpg',
            'data/raw/patient/hypo_002.jpg'
        ]
        self.test_image_healthy = [
            'data/test/healthy/801.jpg',
            'data/test/healthy/802.jpg',
            'data/test/healthy/803.jpg',
            'data/test/healthy/804.jpg'
        ]
        self.test_image_patient = [
            'data/test/patient/1601_l.jpg',
            'data/test/patient/1602_l.jpg',
            'data/test/patient/1603_r.jpg',
            'data/test/patient/1604_r.jpg',
        ]

    def testCreateLoadSaveModel(self):
        failed = False
        try: # test loading models
            for model_name in self.models:
                print(f'\ntest loading model: \"{model_name}\" ...')
                detector = StrabismusDetector(model_name=model_name, debug=False)
                self.assertIsNot(detector.model, None)
                self.assertTrue(detector.model['trained'])
                self.assertNotEqual(len(detector.model['name']), 0)
                self.assertEqual(detector.model['name'], model_name)
                self.assertIsNot(detector.model['model'], None)
                print('PASS!')
            detector = StrabismusDetector(model_name='test_lenet')
            self.assertIs(detector.model, None)
        except Exception as err:
            print(str(err))
            failed = True
        self.assertFalse(failed)

        try: # test creating models
            detector = StrabismusDetector(model_name=self.models[0], debug=False)
            res = detector.create_model(model_name='test_lenet')
            self.assertTrue(res)
            self.assertEqual(detector.model['name'], 'test_lenet')
            self.assertFalse(detector.model['trained'])
            self.assertIsNot(detector.model['model'], None)
        except Exception as err:
            print(str(err))
            failed = True
        self.assertFalse(failed)

        try: # test saving models
            res = detector._save_model_()
            self.assertFalse(res)
            res = detector._save_model_(self.models[0])
            self.assertFalse(res)
        except Exception as err:
            print(str(err))
            failed = True
        self.assertFalse(failed)

    def testTrainModel(self):
        ...

    def testDetection(self):        
        failed = False
        try:
            detector = StrabismusDetector(self.models[2])
            res = detector.isStrabismus(input_image=self.raw_image_healthy[0],
                                        processed=False)
            self.assertFalse(res)
        except Exception as err:
            print(str(err))
            failed = True
        self.assertFalse(failed)
        try:
            res = detector.isStrabismus(input_image=self.raw_image_patient[0],
                                        processed=False)
            self.assertTrue(res)
        except Exception as err:
            print(str(err))
            failed = True
        self.assertFalse(failed)

if __name__ == '__main__':
    unittest.main()
