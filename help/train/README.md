# Train Fast-RCNN on Another Dataset

We will illustrate how to train Fast-RCNN on another dataset in the following steps, and we will take **INRIA Person** as the example dataset.

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
         |-- train.txt
```

The `train.txt` contains all the names(without extensions) of images files that will be used for training. For example, there are a few lines in `train.txt` below.

```
crop_000011
crop_000603
crop_000606
crop_000607
crop_000608
```

### Construct IMDB

You need to add a new python file describing the dataset we will use to the directory `$FRCNN_ROOT/lib/datasets`, see `inria.py`. Then the following steps should be taken.
  - Modify `self._classes` in the constructor function to fit your dataset.
  - Be careful with the extensions of your image files. See `image_path_from_index` in `inria.py`.
  - Write the function for parsing annotations. See `_load_inria_annotation` in `inria.py`.
  - Do not forget to add `import` syntaxes in your own python file and other python files in the same directory.

Then you should modify the `factory.py` in the same directory. For example, to add **INRIA Person**, we should add

```sh
inria_devkit_path = '/home/szy/INRIA'
for split in ['train', 'test']:
    name = '{}_{}'.format('inria', split)
    __sets[name] = (lambda split=split: datasets.inria(split, inria_devkit_path))
```

See the example `inria.py` at https://github.com/EdisonResearch/fast-rcnn/blob/master/lib/datasets/inria.py.

### Run Selective Search 

Modify the matlab file `selective_search.m` in the directory `$FRCNN_ROOT/selective_search`, if you do not have that directory, you could find it at https://github.com/EdisonResearch/fast-rcnn/tree/master/selective_search. 

```sh
image_db = '/home/szy/INRIA/';
image_filenames = textread([image_db '/data/ImageSets/train.txt'], '%s', 'delimiter', '\n');
for i = 1:length(image_filenames)
    if exist([image_db '/data/Images/' image_filenames{i} '.jpg'], 'file') == 2
	image_filenames{i} = [image_db '/data/Images/' image_filenames{i} '.jpg'];
    end
    if exist([image_db '/data/Images/' image_filenames{i} '.png'], 'file') == 2
        image_filenames{i} = [image_db '/data/Images/' image_filenames{i} '.png'];
    end
end
selective_search_rcnn(image_filenames, 'train.mat');
```

Run this matlab file and then move the output `train.mat` to the root directory of your dataset, here it should be `/home/szy/INRIA/`. As it is a time consuming process, please be patient.

### Modify Prototxt

For example, if you want to use the model **VGG_CNN_M_1024**, then you should modify `train.prototxt` in `$FRCNN_ROOTmodels/VGG_CNN_M_1024`, it mainly concerns with the number of classes you want to train. Let's assume that the number of classes is `C (do not forget to count the `background` class). Then you should 
  - Modify `num_classes` to `C`;
  - Modify `num_output` in the `cls_score` layer to `C`
  - Modify `num_output` in the `bbox_pred` layer to `4 * C`

See https://github.com/rbgirshick/fast-rcnn/issues/11 for more details. 

### Train!

In the directory **$FRCNN_ROOT**, run the following command in the shell.

```sh
./tools/train_net.py --gpu 0 --solver models/VGG_CNN_M_1024/solver.prototxt \
    --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel --imdb inria_train
```

Be careful with the **imdb** argument as it specifies the dataset you will train on. Then just drink a cup of coffee and take a break to wait for the training.

### References

[Fast-RCNN] https://github.com/rbgirshick/fast-rcnn
