"""
Project: Detecting Strabismus with Convolutional Neural Networks
File   : ut_dector.py
Purpose: unit testing detector methods
Author : Chuan Zhang
Email  : chuan.zhang2015@gmail.com
Example: $ python -m unittest -v ut_detector.py
"""
import os
import sys
import unittest

pkg_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../'
)
sys.path.insert(0, pkg_dir)

from strabismus.detector import StrabismusDetector

class ModelPredictionTestSuite(unittest.TestCase):
    """
    Purpose: testing Model predictions
    Features to be tested:
        * Create, Save and Load Models
        * Train a given model with given data
        * Make predictions using a given model
    """

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

    def test_create_load_save_model(self):
        """test creating, saving, and loading models"""
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

    def test_train_model(self):
        """test training models"""
        ...

    def test_prediction(self):
        """test making predictions"""
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
