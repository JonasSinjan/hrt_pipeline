"""
test many iterations of input files - so that all combinations are tested
"""
import pytest
import unittest
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../src/')

from sophi_hrt_pipe.hrt_pipe import phihrt_pipe
from sophi_hrt_pipe.utils import *
from sophi_hrt_pipe.inversions import *
from sophi_hrt_pipe.processes import *

def test_one():
  #test with almost everything off after reading in all files
  phihrt_pipe(f"./test_jsons/test_1.json")
  
def test_no_science():
  #test with no science data
  with pytest.raises(ValueError):
    phihrt_pipe(f"./test_jsons/test_2.json")

def test_no_flat():
  #test with no flat data
 with pytest.raises(ValueError):
    phihrt_pipe(f"./test_jsons/test_3.json")

def test_no_dark():
  #test with no science data
 with pytest.raises(ValueError):
    phihrt_pipe(f"./test_jsons/test_4.json")

def test_flat_as_dark():
  #test with dark and flat swapped
 with pytest.raises(ValueError):
    phihrt_pipe(f"./test_jsons/test_5.json")

def test_missing_vtoqu_keyword():
  #test with missing limb keyword
  with pytest.raises(KeyError) as e:
    phihrt_pipe(f"./test_jsons/test_8.json")
    assert 'VtoQU' in e

def test_nonstr_input_for_data():
  #test with a non string input given for data_f, flat_f or dark_f
  pass

  

  
    

