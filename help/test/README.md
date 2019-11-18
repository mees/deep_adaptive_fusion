# Test Fast-RCNN on Another Dataset

We will illustrate how to test Fast-RCNN on another dataset in the following steps, and we will take **INRIA Person** as the example dataset.

### Format Your Dataset

At first, the dataset must be well organzied with the required format.
```
INRIA
|-- data
    |-- Annotations
         |-- *.txt (Annotation files)
    |-- Images
         |-- *.png (Image files)
    |-- ImageSets
         |-- test.txt
|-- results
    |-- test (empty before test)
|-- VOCcode (optical)
```

The `test.txt` contains all the names(without extensions) of images files that will be used for training. For example, there are a few lines in `test.txt` below.

```
crop_000001
crop_000002
crop_000003
crop_000004
crop_000005
```

### Construct IMDB

See it at https://github.com/EdisonResearch/fast-rcnn/tree/master/help/train.

Actually you do not need to implement the `_load_inria_annotation`, you could just use `inria.py` to construct IMDB for your own dataset. For example, to train on a dataset named **TownCenter**, just the followings to `factory.py`.

```sh
towncenter_devkit_path = '/home/szy/TownCenter'
for split in ['test']:
   name = '{}_{}'.format('towncenter', split)
   __sets[name] = (lambda split=split: datasets.inria(split, towncenter_devkit_path))
```

### Run Selective Search 

See it at https://github.com/EdisonResearch/fast-rcnn/tree/master/help/train.

Note that it should be `test.mat` rather than `train.mat`.

### Modify Prototxt

For example, if you want to use the model **VGG_CNN_M_1024**, then you should modify `test.prototxt` in `$FRCNN_ROOTmodels/VGG_CNN_M_1024`, it mainly concerns with the number of classes you want to train. Let's assume that the number of classes is `C (do not forget to count the `background` class). Then you should 
  - Modify `num_output` in the `cls_score` layer to `C`
  - Modify `num_output` in the `bbox_pred` layer to `4 * C`

See https://github.com/rbgirshick/fast-rcnn/issues/11 for more details. 

### Prepare Your Evaluation Code

In the original framework of **Fast-RCNN**, it uses matlab wrappers to evluate the results. As the evluation process is not very difficult, you could modify the function `evaluate_detections` in `inria.py`.  

As **INRIA Person** provides some matlab files in the format of **PASCAL-VOC**, you could modify it a little and use it directly. You could see https://github.com/EdisonResearch/fast-rcnn/tree/master/help/INRIA/VOCcode for the VOCcode.

If you do not want to use the evluation function in the framework of **Fast-RCNN**, you could find the results in the directory `results/test` in the roor directory of your dataset.

### Test!

In the directory **$FRCNN_ROOT**, run the following command in the shell.

```sh
./tools/test_net.py --gpu 1 --def models/VGG_CNN_M_1024/test.prototxt \
    --net output/default/train/vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel --imdb inria_test
```

Be careful with the **imdb** argument as it specifies the dataset you will train on. 

### References

[Fast-RCNN] https://github.com/rbgirshick/fast-rcnn

