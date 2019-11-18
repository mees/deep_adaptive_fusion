# Choosing Smartly: Adaptive Multimodal Fusion for Object Detection in Changing Environments

This is the code implementing an adaptive gating sensor fusion approach for object detection based on a mixture of convolutional neural networks. More information at our project page ```http://adaptivefusion.cs.uni-freiburg.de```

## Reference
If you find the code helpful please consider citing our work 
```
@INPROCEEDINGS{mees16iros,
  author = {Oier Mees and Andreas Eitel and Wolfram Burgard},
  title = {Choosing Smartly: Adaptive Multimodal Fusion for Object Detection in Changing Environments},
  booktitle = {Proceedings of the International Conference on Intelligent Robots and Systems (IROS)},
  year = 2016,
  address = {Daejeon, South Korea},
  url = {http://ais.informatik.uni-freiburg.de/publications/papers/mees16iros.pdf},
}
```

## Installation
Please refer to [INSTALL.md](INSTALL.md) for setup instructions. The code was tested on Ubuntu 14.04. 
The gating layer implementing the adaptive fusion scheme can be found at ```caffe-fast-rcnn/src/caffe/layers/gating_inner_product_layer.cpp and gating_inner_product_layer.cu ```

## Models
We provide several models from the paper. A RGB-D gating network trained on the InOutDoorPeople dataset is available at ```models/googlenet_rgb_depth_gating_All_Dropout/googlenet_rgb_depth_gating_iter_2500.caffemodel```.


## Dataset
Our  InOutDoorPeople dataset containing 8305 annotated frames of RGB and Depth data can be found at ```http://adaptivefusion.cs.uni-freiburg.de/#dataset```
