"""
test many iterations of input files - so that all combinations are tested
"""
import pytest
import unittest
import numpy as np
import sys
sys.path.append('../')

from src.hrt_pipe import phihrt_pipe

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

def test_missing_limb_keyword():
  with pytest.raises(KeyError):
    phihrt_pipe(f"./test_jsons/test_8.json")

  

  
    

