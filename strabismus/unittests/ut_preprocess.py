import os
import sys
import unittest

pkg_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../'
)
sys.path.insert(0, pkg_dir)

from detector import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH
)

from detector.preprocess import (
    Vertex,
    Region,
    Image
)


class TestSuite_Preprocessing(unittest.TestCase):
    """
    Purpose: testing preprocessing code
    Data Structures to be tested:
        * Vertex
        * Region
        * Image
    Features to be tested:
        * Load Image
        * Crop Image
    """

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

    def test_Vertex(self):
        '''testing Vertex data structure'''
        p = Vertex()
        self.assertEqual(p.x, 0)
        self.assertEqual(p.y, 0)
        p = Vertex(-1000, 235)
        self.assertEqual(p.x, -1000)
        self.assertEqual(p.y, 235)
        self.assertEqual(str(p), '(-1000, 235)')
        try:
            p.x =  200
            p.y = -134
        except Exception as err:
            self.failureException(err)
        self.assertEqual(p.x,  200)
        self.assertEqual(p.y, -134)
        self.assertEqual(str(p), '(200, -134)')
        
    def test_Region(self):
        r = Region()
        self.assertEqual(str(r), '(0, 0, 0, 0)')
        self.assertEqual(r.height, 0)
        self.assertEqual(r.width, 0)
        self.assertTrue(r.is_empty())
        # left < 0
        try:
            r.setRegion(Vertex(-1, 2), Vertex(3, 4))
        except ValueError as ve:
            print(str(ve))
        else:
            self.fail('ValueError is expected!')
        # top < 0
        try:
            r.setRegion(Vertex(1, -2), Vertex(3, 4))
        except ValueError as ve:
            print(str(ve))
        else:
            self.fail('ValueError is expected!')
        # right < 0
        try:
            r.setRegion(Vertex(1, 2), Vertex(-3, 4))
        except ValueError as ve:
            print(str(ve))
        else:
            self.fail('ValueError is expected!')
        # bottom < 0
        try:
            r.setRegion(Vertex(1, 2), Vertex(3, -4))
        except ValueError as ve:
            print(str(ve))
        else:
            self.fail('ValueError is expected!')
        # top > bottom : top is below bottom
        try:
            r.setRegion(Vertex(1, 4), Vertex(3, 2))
        except ValueError as ve:
            print(str(ve))
        else:
            self.fail('ValueError is expected!')
        # left > right
        try:
            r.setRegion(Vertex(3, 4), Vertex(1, 2))
        except ValueError as ve:
            print(str(ve))
        else:
            self.fail('ValueError is expected!')

        try:
            r1 = Region(Vertex(1, 10), Vertex(18, 20))
            r2 = Region(Vertex(2, 16), Vertex(15, 18))
            r3 = Region(Vertex(3, 12), Vertex(19, 17))
        except:
            self.fail('Region initialization error!')
        self.assertEqual(r1.height, 10)
        self.assertEqual(r3.width, 16)
        self.assertTrue(r1.contains(r2))
        self.assertFalse(r1.contains(r3))
        self.assertFalse(r2.contains(r3))

        r2.union(r3)
        self.assertTrue(r2.contains(r3))
        self.assertEqual(str(r2), '(2, 12, 19, 18)')
        r1.intersect(r3)
        self.assertTrue(r3.contains(r1))
        
        try:
            r3.shift_vert(-3)
        except:
            self.fail('shift_vert() error!')
        self.assertEqual(str(r3), '(3, 9, 19, 14)')
        self.assertFalse(r2.contains(r3))
        try:
            r3.shift_vert(-10)
        except:
            print('ValueError is expected')
        else:
            self.fail('ValueError is expected!')
        
        try:
            r3.shift_vert(4)
        except:
            self.fail('shift_vert() error!')
        self.assertEqual(str(r3), '(3, 13, 19, 18)')
        self.assertTrue(r2.contains(r3))
        try:
            r3.shift_hori(1)
        except:
            self.fail('shift_hori() error!')
        self.assertEqual(str(r3), '(4, 13, 20, 18)')
        self.assertFalse(r2.contains(r3))

        try:
            r3.shift_hori(-5)
        except:
            print('ValueError is expected')
        else:
            self.fail('ValueError is expected!')
        self.assertEqual(str(r3), '(4, 13, 20, 18)')


if __name__ == '__main__':
    unittest.main()
