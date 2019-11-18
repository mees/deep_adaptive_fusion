import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np
 
# ~-~-~  ~-~-~ ~-~-~ ~-~-~ ~-~-~ ~-~-~ ~-~-~ ~-~-~ 
# ~-~-~  ~-~-~ ~-~-~ ~-~-~ ~-~-~ ~-~-~ ~-~-~ ~-~-~ 
# ~-~-~  ~-~-~ ~-~-~ ~-~-~ ~-~-~ ~-~-~ ~-~-~ ~-~-~ 

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Compare caffemodels')
  parser.add_argument("d1", help="deplot file1")
  parser.add_argument("d2", help="deploy file2")
  parser.add_argument("m1", help="caffemodel1")
  parser.add_argument("m2", help="caffemodel2")

  args = vars(parser.parse_args())
  deploy1 = str(args['d1'])
  deploy2 = str(args['d2'])
  model1 = str(args['m1'])
  model2 = str(args['m2'])
  
  net1 = caffe.Net(deploy1, model1, caffe.TEST)
  net2 = caffe.Net(deploy2,model2, caffe.TEST)
  
  for param in net1.params:
      print param
      param_net2 = param + '_NET1'
      a1 =  net1.params[param][1].data
      a2 =  net2.params[param_net2][1].data 
      if np.array_equal(a1,a2):
          print ' equal'
      else:
          print 'unequal'
          print param, param_net2
          print a1
          print a2 

