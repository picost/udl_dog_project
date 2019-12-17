#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:18:45 2019

@author: picost
"""

import unittest
from helpers import conv2d_carac

class HelpTest(unittest.TestCase):
    
    def test_01(self):
       h_out, w_out, n_features, n_params = conv2d_carac(32, 32, 3, 16, 3, 1, 1)
       self.assertEqual(h_out, 32)
       self.assertEqual(w_out, 32)
       self.assertEqual(n_features, 32 * 32 * 16)
       self.assertEqual(n_params, 3**2 * 3 * 16)


#suite = unittest.TestSuite()
#suite.addTest(HelpTest())

if __name__ == '__main__':
    unittest.main()