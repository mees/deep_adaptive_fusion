# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.inria
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import yaml
import cv2

class inria(datasets.imdb):
    def __init__(self, image_set, devkit_path):
        datasets.imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._roidb_txtfiles_path = os.path.join(self._devkit_path, 'roidb_rgb_sliding')
        self._path_suffix = 'ImagesQhd'
        self._num_classes = 2
        self._classes = ('__background__', # always index 0
                         'person')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.png']
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        if (self._roidb_txtfiles_path):
            self._roidb_handler = self.txtfiles_roidb
        else:
            self._roidb_handler = self.selective_search_roidb

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : -1}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, self._path_suffix,
                                  index + ext)
            image_path = os.path.join(self._data_path, self._path_suffix,
                                  index + ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
	return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb


        gt_roidb = [self._load_inria_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
                ct = 0
                num_pos = 0
                IoU_tresh = 0.6
                for index in self.image_index:
                    im_name = os.path.abspath(os.path.join(self._devkit_path,
                                                'data/Images/' + index + '_rotated.png'))
                    im = cv2.imread(im_name, -1)
                    gt_overlaps = roidb[ct]['gt_overlaps'].toarray()
                    overlap_indx = np.where(gt_overlaps[:,1] > IoU_tresh)
                    for gi in overlap_indx[0]:
                        #print int(roidb[ct]['boxes'][gi])
                        xmin = int(roidb[ct]['boxes'][gi][0])
                        ymin = int(roidb[ct]['boxes'][gi][1])
                        xmax = int(roidb[ct]['boxes'][gi][2])
                        ymax = int(roidb[ct]['boxes'][gi][3])
                        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255,255,0), thickness=10, lineType=8)
                        num_pos += 1
                    oname = '/tmp/gt_overlap_' + index + '_' + '.png'
                    #cv2.imwrite(oname, im)
                    ct +=1;
            print 'Total number of positive training examples including GT with overlap > ', IoU_tresh, ': ' , num_pos
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

 
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)

        else:
            roidb = self._load_selective_search_roidb(None)
	with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)
        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self._devkit_path,
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['all_boxes'].ravel()
        top_k = self.config['top_k']        
        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:top_k, (1, 0, 3, 2)] - 1)

	return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 self.name))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def txtfiles_roidb(self):
        """
        return the database of regions of interest from textfile format:
        xmin ymin xmax ymax
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_txtfiles_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        gt_roidb = self.gt_roidb()
        # Remove frames with no object annotations
        orig_image_index = self.image_index
        removeset = set()

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} textfiles roidb loaded from {}'.format(self.name, cache_file)
            if self._image_set == 'train':
                for index in xrange(len(gt_roidb)):
                    if gt_roidb[index] is None:
                        removeset.add(index)
                mod_image_index =  [v for i, v in enumerate(orig_image_index) if i not in removeset]
                del self.image_index[:]
                for index in mod_image_index:
                    self.image_index.append(index)
            return roidb

        if self._image_set == 'train':
            ss_roidb = self._load_txtfiles_roidb(gt_roidb)
            for index in xrange(len(gt_roidb)):
                if gt_roidb[index] is None:
                    removeset.add(index)
            mod_image_index =  [v for i, v in enumerate(orig_image_index) if i not in removeset]
            del self.image_index[:]
            for index in mod_image_index:
                self.image_index.append(index)
            print "Ignoring frames with no object annotations"
            print "len gt roidb", len(gt_roidb), "len ss roidb", len(ss_roidb), "len image index", len(self.image_index)
            gt_roidb = [index for index in gt_roidb if index is not None]
            ss_roidb = [index for index in ss_roidb if index is not None]
            print "len gt roidb", len(gt_roidb), "len ss roidb", len(ss_roidb), "len image index", len(self.image_index)
            assert len(gt_roidb) == len(ss_roidb) == len(self.image_index)

            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        elif self._image_set == 'val':
            roidb = self._load_txtfiles_roidb(None)
            "len roidb", len(roidb), "len image index", len(self.image_index)
            assert len(roidb) == len(self.image_index)
        else: 
            roidb = self._load_txtfiles_roidb(None)
            "len roidb", len(roidb), "len image index", len(self.image_index)
            assert len(roidb) == len(self.image_index)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote textfiles roidb to {}'.format(cache_file)

        return roidb

    def _load_txtfiles_roidb(self, gt_roidb):
        assert os.path.exists(self._roidb_txtfiles_path), \
               'Textfiles data not found at: {}'.format(self._roidb_txtfiles_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(self._roidb_txtfiles_path, self.image_index[i] + '.txt')
            raw_data = np.loadtxt(filename)
            if(top_k == -1):
                box_list.append((raw_data).astype(np.uint16))
            else:
                box_list.append((raw_data[:top_k]).astype(np.uint16))
        return self.create_roidb_from_box_list(box_list, gt_roidb)



    # def _load_inria_annotation(self, index):
    #     """
    #     Load image and bounding boxes info from txt files of INRIAPerson.
    #     """
    #     filename = os.path.join(self._data_path, 'Annotations', index + '.txt')
    #     # print 'Loading: {}'.format(filename)
    # 	with open(filename) as f:
    #             data = f.read()
    # 	import re
    # 	objs = re.findall('\(\d+, \d+\)[\s\-]+\(\d+, \d+\)', data)

    #     num_objs = len(objs)

    #     boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    #     gt_classes = np.zeros((num_objs), dtype=np.int32)
    #     overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

    #     # Load object bounding boxes into a data frame.
    #     for ix, obj in enumerate(objs):
    #         # Make pixel indexes 0-based
	   #  coor = re.findall('\d+', obj)
    #         x1 = float(coor[0])
    #         y1 = float(coor[1])
    #         x2 = float(coor[2])
    #         y2 = float(coor[3])
    #         cls = self._class_to_ind['person']
    #         boxes[ix, :] = [x1, y1, x2, y2]
    #         gt_classes[ix] = cls
    #         overlaps[ix, cls] = 1.0

    #     overlaps = scipy.sparse.csr_matrix(overlaps)

    #     return {'boxes' : boxes,
    #             'gt_classes': gt_classes,
    #             'gt_overlaps' : overlaps,
    #             'flipped' : False}

    def _load_inria_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of INRIAPerson.
        """

        filename = os.path.join(self._data_path, 'Annotations', index + '.yml')
        stream = open(filename,'r')
        #hack to skip the ehader and read yml correctly
        stream.seek(9)
        newyaml=yaml.load(stream)
        if 'object' not in newyaml['annotation']:
            print "empty yml ", filename[:-4]
            return None
        num_objs = len(newyaml['annotation']['object'])
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        num_objs = len(newyaml['annotation']['object'])
        # Load object bounding boxes into a data frame.
        for ix in range(0,num_objs):
            x1 = int(newyaml['annotation']['object'][ix]['bndbox']['xmin'])/2
            y1 = int(newyaml['annotation']['object'][ix]['bndbox']['ymin'])/2
            x2 = int(newyaml['annotation']['object'][ix]['bndbox']['xmax'])/2
            y2 = int(newyaml['annotation']['object'][ix]['bndbox']['ymax'])/2
            cls = self._class_to_ind['person']
            boxes[ix, :] = [int(x1), int(y1), int(x2), int(y2)]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _write_inria_results_file(self, all_boxes, output_dir):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/comp4-44503_det_test_aeroplane.txt
        #path = os.path.join(self._devkit_path, 'results', self.name, comp_id + '_')
        path = os.path.join(output_dir,comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            #print 'Writing {} results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            print 'Writing results file ', filename
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        if(k == 0):
                            f.write('{:s} {:d} {:d} {:d} {:d} {:.6f};'.
                                format(index, int(dets[k, 0]) + 1, int(dets[k, 1]) + 1,
                                        int(dets[k, 2]) + 1, int(dets[k, 3]) + 1, float(dets[k, 4])))
                        else:
                            f.write(' {:d} {:d} {:d} {:d} {:.6f};'.
                                format( int(dets[k, 0]) + 1, int(dets[k, 1]) + 1,
                                        int(dets[k, 2]) + 1, int(dets[k, 3]) + 1, float(dets[k, 4]))) 
                    f.write('\n')        
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'setenv(\'LC_ALL\',\'C\'); voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_inria_results_file(all_boxes, output_dir)
        #self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.inria('train', 'test')
    res = d.roidb
    from IPython import embed; embed()
