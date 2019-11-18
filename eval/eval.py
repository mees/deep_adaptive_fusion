# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:35:39 2016

@author: meeso
"""
import argparse
import errno
import json
import operator
import os
import sys
from math import cos, sin
from os.path import join, expanduser

import cv2
import numpy as np
import pylab as pl
import time
import yaml
import matplotlib.pyplot as plt

import ray
from voc_ap import voc_ap
from visualization.uncertainty import plotUncertaintyGraphs
from visualization.gating import plot_gating_values
from iou import interUnio, bb_intersection_over_union

BASEDIR = '/home/meyerjo/dataset/inoutdoorpeoplergbd/'
DIR_GT = BASEDIR + '/Annotations/'
testBBDict = {}
# gtBBDict = {}
boxes_pred_dict = {}
sortedListTestBB = []
noDetectionList = []
visualOutput = False
plotGraphs = False
# numOfGTBBs = 0
SOFTMAX_THRESHOLD = None
OUTPUT_PATH = None
im_suffix = ['ImagesQhd', 'DepthJetQhd']


def writeGroundTruth(output_file):
    # Comment this in if you want to create the gt_dict.txt for the first time
    count = 1
    gtBBDict = {}
    for filename in os.listdir(DIR_GT):
        input_filename = DIR_GT + filename[:-4] + '.yml'
        if not os.path.isfile(input_filename):
            print('yml not found for', filename[:-4])
            if filename[:-4] not in gtBBDict:
                gtBBDict[filename[:-4]] = []
            continue
        stream = open(input_filename, 'r')
        # hack to skip the ehader and read yml correctly
        stream.seek(9)
        newyaml = yaml.load(stream)
        if 'object' not in newyaml['annotation']:
            print("empty yml ", filename[:-4])
            if filename[:-4] not in gtBBDict:
                gtBBDict[filename[:-4]] = []
            continue
        for i in range(0, len(newyaml['annotation']['object'])):
            xmin = int(int(newyaml['annotation']['object'][i]['bndbox']['xmin']) / 2)
            ymin = int(int(newyaml['annotation']['object'][i]['bndbox']['ymin']) / 2)
            xmax = int(int(newyaml['annotation']['object'][i]['bndbox']['xmax']) / 2)
            ymax = int(int(newyaml['annotation']['object'][i]['bndbox']['ymax']) / 2)
            if filename[:-4] not in gtBBDict:
                gtBBDict[filename[:-4]] = [[xmin, ymin, xmax, ymax]]
            else:
                gtBBDict[filename[:-4]].append([xmin, ymin, xmax, ymax])
    json.dump(gtBBDict, open(output_file, 'w'))
    return gtBBDict


def readGroundTruthValBB():
    # Comment this in if you want to create the gt_dict.txt for the first time
    groundtruth_dict_path = './gt_dict.txt'
    #
    # global gtBBDict
    gtBBDict = None
    if os.path.exists(groundtruth_dict_path):
        gtBBDict = json.load(open(groundtruth_dict_path))
    else:
        print('Groundtruth dictionary is missing. Writing the file is deactivated')
        # writeGroundTruth(groundtruth_dict_path)
    return gtBBDict


def readTestSet(groundTruthObject):
    numOfGTBBs = 0
    with open(DIR_TESTSET, 'r') as stream:
        num_frames = 0
        for line in stream:
            numOfGTBBs = numOfGTBBs + len(groundTruthObject[line.strip()])
            num_frames += 1
        print("Number of GTBBs,\t{0}".format(numOfGTBBs))
        print("Number of test frames,\t{0}".format(num_frames))
    return numOfGTBBs


def enterResultingBBsinDict(filename, xmin, ymin, xmax, ymax, score):
    if filename not in testBBDict:
        testBBDict[filename] = [score, [xmin, ymin, xmax, ymax]]
    else:
        testBBDict[filename].append([score, [xmin, ymin, xmax, ymax]])


def get_annotations(file_name, groundtruthboxes_dict):
    imgsNoPerson = []
    with open(file_name) as f:
        for ln in f:
            imgName = ln.split()[0]
            if imgName not in groundtruthboxes_dict:
                imgsNoPerson.append(imgName)
    return imgsNoPerson


def get_results_array(DIR_IN):
    result = []
    with open(DIR_IN) as f:
        for line in f:
            original_line, gating_values = line, '0,0'
            if '|' in line:
                original_line, gating_values = line.split('|')
            gating_factor_1, gating_factor_2 = gating_values.split(',')
            gating_factor_1, gating_factor_2 = float(gating_factor_1), float(gating_factor_2)
            splittedLine = original_line.split()
            imgName = splittedLine[0]
            bboxes = splittedLine[1:]
            result.append(
                [imgName, bboxes, [gating_factor_1, gating_factor_2]]
            )
    return result


def readAndSortBBs(DIR_IN, groundtruth_boxes):
    # global numOfGTBBs
    imgsNoPerson = get_annotations(DIR_IN, groundtruth_boxes)
    lines_boxes = get_results_array(DIR_IN)

    for res in lines_boxes:
        imgName = res[0]
        bboxes = res[1]
        gating_factor_1, gating_factor_2 = res[2][0], res[2][1]

        # imgName = splittedLine[0][:-16] # for rgb -4, else 16, 0 for rcnn
        if imgName in imgsNoPerson:
            print('No person annotation in: ', imgName)
            pass

        if len(bboxes) == 0:
            noDetectionList.append(imgName)
        for i in xrange(0, len(bboxes), 5):
            tmp_softmax_value = float(bboxes[i + 4][:-1])
            #  Johannes: changed threshold as gating leads to softmax values of 0.5
            if tmp_softmax_value > SOFTMAX_THRESHOLD:
                tmp_entry = [str(0.0), imgName, bboxes[i + 1], bboxes[i], bboxes[i + 3], bboxes[i + 2]]
                # This normalization is only required as we compute our bounding boxes on the full-hd resolution
                tmp_entry[2] = str(int(int(tmp_entry[2]) / 2.))
                tmp_entry[3] = str(int(int(tmp_entry[3]) / 2.))
                tmp_entry[4] = str(int(int(tmp_entry[4]) / 2.))
                tmp_entry[5] = str(int(int(tmp_entry[5]) / 2.))

                tmp_entry += [res[2][0], res[2][1]]

                if tmp_softmax_value >= 0.01:
                    tmp_entry[0] = tmp_softmax_value
                sortedListTestBB.append(tuple(tmp_entry))

    return sorted(sortedListTestBB, key=operator.itemgetter(0), reverse=True), imgsNoPerson


def plot_precision_recall(precisionList, recallList):
    pl.clf()
    pl.xlabel('Recall', fontsize=18)
    pl.ylabel('Precision', fontsize=18)
    pl.ylim([0.0, 1.0])
    pl.xlim([0.0, 1.0])
    pl.plot(recallList, precisionList, lw=1)


def frange(start, stop, step):
    x = start
    while x < stop:
        yield x
        x += step


def compute_mAP(precisionList, recallList):
    # compute average precision
    ap = 0
    p = 0
    precs = np.array(precisionList)
    recs = np.array(recallList)
    for t in frange(0, 1.0, 0.1):
        p = 0
        if any(precs[np.where(recs >= t)]):
            p = max(precs[np.where(recs >= t)])
        ap = ap + p / 11
    return ap


def writeGT(groundtruthDict):
    if not os.path.exists(BASEDIR + '/results/images_gt/'):
        os.mkdir(BASEDIR + '/results/images_gt/')
    for key in groundtruthDict:
        fname = BASEDIR + '/data/' + im_suffix[0]
        img2 = cv2.imread(fname + '/' + key + '.png', 1)
        for indice, (xmin, ymin, xmax, ymax) in enumerate(groundtruthDict[key]):
            cv2.rectangle(img2, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 255), 2)
        cv2.imwrite(BASEDIR + '/results/images_gt/' + key + '.png', img2)


def createDictFromPred(boxes_pred):
    if not os.path.exists(boxes_pred):
        print('Prediction file does not exists')
        exit(-1)
    lines_boxes = get_results_array(boxes_pred)
    boxes_pred_dict = {}
    gating_factors = []
    file_names = []
    for res in lines_boxes:
        imgName = res[0]
        bboxes = res[1]
        gating_factor_1, gating_factor_2 = res[2][0], res[2][1]
        gating_factors.append([gating_factor_1, gating_factor_2])
        file_names.append(imgName)

        if imgName in boxes_pred_dict:
            print("Error: duplicate frame in detection output", imgName)
            exit(1)

        for i in xrange(0, len(bboxes), 5):
            # tmp_array = [splittedLine[i], splittedLine[i + 1], splittedLine[i + 2], splittedLine[i + 3]]
            # print(splittedLine, len(splittedLine))
            if float(bboxes[i + 4][:-1]) < SOFTMAX_THRESHOLD:  # ignore those entries
                continue

            tmp_array = [bboxes[i], bboxes[i + 1], bboxes[i + 2], bboxes[i + 3]]
            # normalize coordinated to FullHd / 2
            tmp_array = np.asarray(list(map(int, tmp_array))) / 2.
            tmp_array = list(np.floor(tmp_array).astype(int).tolist())
            # switch the axis - REQUIRED because of the current output format (2018 - 02 - 06)
            tmp_array = [tmp_array[1], tmp_array[0], tmp_array[3], tmp_array[2]]
            # back to string
            tmp_array = list(map(str, tmp_array))
            tmp_array += [gating_factor_1, gating_factor_2]

            if imgName not in boxes_pred_dict:
                boxes_pred_dict[imgName] = [tmp_array]
            else:
                boxes_pred_dict[imgName].append(tmp_array)

    file_names_ind = np.argsort(np.asarray(file_names))
    gating_factors = np.asarray(gating_factors)
    gating_factors = gating_factors[file_names_ind, :]
    # global boxes_pred_dict
    return boxes_pred_dict, gating_factors


def load_img_predictions(basedir, image_suffixes, imgName):
    imgPreds = {}
    for suffix in image_suffixes:
        imPath = basedir + suffix + '/' + imgName + '.png'
        if not os.path.exists(imPath):
            print('Image-File does not exist: {0}'.format(imPath))
            exit(-1)
        imgPred = cv2.imread(imPath, -1)
        imgPreds[suffix] = imgPred
    return imgPreds


def mkdir_p(path):
    """
    mkdir recursive ("mkdir -p")
    :param path:
    :return:
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def draw_boxes(boxes_pred_dict, subpath=None, output=None):
    """
    Outputs the images from all modalities with the corresponding
    :param boxes_pred_dict:
    :param subpath:
    :param output:
    :return:
    """
    if output is None:
        opath = BASEDIR + '/results/cache/images'
    else:
        opath = output + '/results/cache/images'
    if subpath is not None and isinstance(subpath, str):
        opath = join(opath, subpath)
    if not os.path.exists(opath):
        mkdir_p(opath)
    stream = open(DIR_TESTSET, 'r')
    frame_no = 0
    for imgName in stream:
        imgName = imgName.strip()
        imgPreds = load_img_predictions(BASEDIR, im_suffix, imgName)
        imgInference = None
        input_filename_groundtruth = join(BASEDIR, 'ImagesQhd', (imgName + '.png'))
        if not os.path.exists(input_filename_groundtruth):
            print('Files does not exist: {0}'.format(input_filename_groundtruth))
            continue
        imgGT = cv2.imread(input_filename_groundtruth, -1)

        if imgName in boxes_pred_dict:
            imgInference = boxes_pred_dict[imgName]

        cv2.putText(imgGT, 'Frame: {0}'.format(str(frame_no)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        cv2.putText(imgGT, 'Frame: {0}'.format(str(frame_no)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        if imgInference is not None and len(imgInference) > 0:
            # cv2.fillPoly(imgGT, np.asarray(
            #     [[[50, 150]], [[50, 350]], [[250, 150]], [[250, 350]]], dtype=np.int32),
            #              color=(255, 0, 0))
            # cv2.fillPoly(imgGT, np.asarray([[[0, 0]], [[50, 0]], [[0, 50]], [[50, 50]]], dtype=np.int32) + 500,
            #              color=(255, 0, 0))
            cv2.rectangle(imgGT, (50, 175), (250, 300), (0, 0, 0), -1, 8)
            cv2.putText(imgGT, 'RGB: {0:.4f}'.format(imgInference[0][-2]),
                        (50, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
            cv2.putText(imgGT, '  D: {0:.4f}'.format(imgInference[0][-1]),
                        (50, 250), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))

        # Write the Image box to the images
        if imgInference is not None:
            for indice, (xmin, ymin, xmax, ymax, gf1, gf2) in enumerate(imgInference):
                for suffix in im_suffix:
                    cv2.rectangle(imgPreds[suffix], (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 255), 2)
        concat_img = None
        if len(im_suffix) == 1:
            concat_img = np.concatenate((imgPreds[im_suffix[0]], imgGT), axis=1)
        elif len(im_suffix) == 2:
            concat_img = np.concatenate((imgPreds[im_suffix[0]], imgPreds[im_suffix[1]], imgGT), axis=1)
        elif len(im_suffix) == 3:
            concat_img = np.concatenate(
                (imgPreds[im_suffix[0]], imgPreds[im_suffix[1]], imgPreds[im_suffix[2]], imgGT), axis=1)
        res_path = opath + '/res_' + imgName + '.png'

        if concat_img is not None:
            cv2.imwrite(res_path, concat_img)
        frame_no += 1
    stream.close()


def visualize_output(input_path, bbox, output_file):
    """
    Cuts the part described in the bbox from the image and writes it to the output_file
    :param input_path:
    :param bbox:
    :param output_file:
    :return:
    """
    if not os.path.exists(input_path):
        print('Input path does not exist: {0}'.format(input_path))
        return
    ymin = int(bbox[0])
    ymax = int(bbox[1])
    xmin = int(bbox[2])
    xmax = int(bbox[3])
    if ymin == ymax:
        print('Image has zero height')
        return
    if xmin == xmax:
        print('Image has zero width')
        return
    input_image = cv2.imread(input_path, -1)
    image_part = input_image[int(ymin):int(ymax), int(xmin):int(xmax)]
    # print "False positive at ", iname
    if len(image_part) == 0:
        print('Image is empty')
        return
    cv2.imwrite(output_file, image_part)

def f1_score(precision, recall):
    return 2 * ((precision* recall)/(precision+recall))


def evaluate(sorted_dict, groundtruth, number_of_groundtruth_boxes, threshold=0.6, is_fullhd=True):
    print('IoU-Threshold,\t{0}'.format(threshold))
    print('Softmax-Threshold,\t{0}'.format(SOFTMAX_THRESHOLD))
    tp = 0
    fp = 0
    fn = 0
    precisionList = []
    recallList = []
    true_positive_difference = []
    iou_true_positives = []

    val_diff = []

    for indice, obj in enumerate(sorted_dict):
        if len(obj) == 6:
            score, img_name, xmin, ymin, xmax, ymax = obj
        elif len(obj) == 8:
            score, img_name, xmin, ymin, xmax, ymax, gating_factor_1, gating_factor_2 = obj
        else:
            print('len(obj) not known')
            break

        testBB = [int(xmin), int(ymin), int(xmax), int(ymax)]
        if img_name in groundtruth:
            # get the groundtruth boxes
            gtBB = groundtruth[img_name]
            for i in range(0, len(gtBB)):
                val_old = interUnio(gtBB[i], testBB)
                val = bb_intersection_over_union(gtBB[i], testBB)
                if val != val_old:
                    val_diff.append(val-val_old)
                if val >= threshold:
                    tp = tp + 1
                    true_positive_difference.append(
                        np.array(gtBB[i]) - np.array(testBB)
                    )
                    iou_true_positives.append(val)
                    # remove the box as we have found it
                    del gtBB[i]
                    break
            else:  # for loop fell through
                fp = fp + 1
                if visualOutput:
                    iname = os.path.join(BASEDIR, im_suffix[0], (img_name + '.png'))
                    tmp_output = OUTPUT_PATH + '/results/cache/fp/' + img_name + str(indice) + '.png'
                    visualize_output(iname, [ymin, ymax, xmin, xmax], tmp_output)
            if (tp + fp) == 0:  # if first matched person is occluded
                print('hack, fixme, first matched person is occluded')
                continue
        else:
            if visualOutput:
                iname = os.path.join(BASEDIR, im_suffix[0], (img_name + '.png'))
                tmp_output = OUTPUT_PATH + '/results/cache/fp/' + img_name + str(indice) + '.png'
                visualize_output(iname, [ymin, ymax, xmin, xmax], tmp_output)
            fp = fp + 1
        precisionList.append(float(tp) / float((tp + fp)))
        recallList.append(float(tp) / float(number_of_groundtruth_boxes))

    testset = open(DIR_TESTSET, 'r')
    # Write false negatives, no detections
    for img_name in testset:
        img_name = img_name.strip()
        if groundtruth[img_name]:
            fn += len(groundtruth[img_name])
            if visualOutput:
                iname = os.path.join(BASEDIR, im_suffix[0], (img_name + '.png'))
                for indice, (xmin, ymin, xmax, ymax) in enumerate(groundtruth[img_name]):
                    tmp_output = OUTPUT_PATH + '/results/cache/fn/' + img_name + str(indice) + '.png'
                    visualize_output(iname, [ymin, ymax, xmin, xmax], tmp_output)
    testset.close()

    print("tp,\t{0}".format(tp))
    print("fp,\t{0}".format(fp))
    print("fn,\t{0}".format(fn))
    print("precision,\t{0}".format(precisionList[-1]))
    print("recall,\t{0}".format(recallList[-1]))
    print("f1 score,\t{0}".format(f1_score(precisionList[-1], recallList[-1])))
    print('avg iou tp,\t{0}'.format(np.mean(iou_true_positives)))

    if plotGraphs:
        plot_precision_recall(precisionList, recallList)
    eer_x = None
    eer_y = None
    for i in range(0, len(precisionList) - 1):
        dist = ray.intersection_dist(recallList[i], precisionList[i] + 0.000001, recallList[i + 1],
                                     precisionList[i + 1], 0.0, 0.0, 1.0, 1.0)
        if dist:
            eer_x = sin(np.pi / 4) * dist
            eer_y = cos(np.pi / 4) * dist
            print('EER:, {0}, {1}'.format(eer_x, eer_y))
            break
    else:
        print('EER:, computation did not find an intersection')

    ap_voc_2007 = voc_ap(np.asarray(recallList), np.asarray(precisionList), True)
    ap_voc_2010 = voc_ap(np.asarray(recallList), np.asarray(precisionList), False)
    m_ap = compute_mAP(precisionList, recallList)
    print('Medium average Precision (Legacy),\t{0}'.format(m_ap))
    # Recall at AP value, we get the index of the element
    recall_map_voc_2007 = min(range(len(precisionList)), key=lambda i: abs(precisionList[i] - ap_voc_2007))
    recall_map_voc_2010 = min(range(len(precisionList)), key=lambda i: abs(precisionList[i] - ap_voc_2010))
    print('Medium average Precision (VOC2007),\t{0}'.format(ap_voc_2007))
    print('Recall at maP (VOC2007),\t{0}'.format(recallList[recall_map_voc_2007]))
    print('Medium average Precision (VOC2010),\t{0}'.format(ap_voc_2010))
    print('Recall at maP (VOC2010),\t{0}'.format(recallList[recall_map_voc_2010]))


    if true_positive_difference is not None:
        true_positive_difference = np.array(true_positive_difference)
        plotUncertaintyGraphs(true_positive_difference)

    if plotGraphs and (eer_x is not None and eer_y is not None):
        pl.plot(eer_x, eer_y, 'ro')
        pl.annotate('EER', xy=(eer_x + 0.01, eer_y))
        x_vect = np.arange(0, 1, 0.1)
        pl.plot(x_vect, x_vect)
        pl.savefig(OUTPUT_PATH + '/results/cache' + '/pr.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset directory')
    parser.add_argument('--boundingboxes', help='Files in which all bounding boxes are aggregated')
    parser.add_argument('--test_file_sequence', help='Sequence of files to use')
    parser.add_argument('--visualize', default=False, help='Visualize the results', dest='visualize',
                        action='store_true')
    parser.add_argument('--picture_output', default=None, type=str, help='Output pictures')
    parser.add_argument('--graph', default=False, help='Plot the graphs', dest='graph', action='store_true')
    parser.add_argument('--softmax_threshold', type=float, default=.5)
    parser.add_argument('--iou_threshold', type=float, default=.6)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    if args.dataset_path is not None:
        BASEDIR = args.dataset_path
    else:
        BASEDIR = os.path.join(expanduser('~'), 'dataset/inoutdoorpeoplergbd')
    DIR_GT = BASEDIR + '/Annotations/'

    test_file_sequence = args.test_file_sequence
    DIR_IN = os.path.abspath(str(args.boundingboxes))
    DIR_TESTSET = os.path.join(BASEDIR, "ImageSets/" + test_file_sequence)
    visualOutput = args.visualize
    plotGraphs = args.graph
    OUTPUT_PATH = None if not visualOutput else args.picture_output
    SOFTMAX_THRESHOLD = args.softmax_threshold

    if visualOutput and OUTPUT_PATH is None:
        raise BaseException('Output Path does not exist')

    if visualOutput and not os.path.exists(OUTPUT_PATH):
        mkdir_p(OUTPUT_PATH)

    print('Evaluating the results')
    print('Input: {0}'.format(DIR_IN))
    print('Test-Sequence: {0}'.format(test_file_sequence))
    print('BaseDir: {0}'.format(BASEDIR))
    print('OutputDir: {0}'.format(OUTPUT_PATH))
    print('Test-Sequence path: {0}'.format(DIR_TESTSET))
    print('Visualizing: {0}'.format('yes' if args.visualize else 'no'))

    # Created directories for result files
    if visualOutput:
        opath = os.path.join(OUTPUT_PATH, 'results/cache/images')
        if not os.path.exists(opath):
            mkdir_p(opath)
        fppath = os.path.join(OUTPUT_PATH, 'results/cache/fp')
        if not os.path.exists(fppath):
            mkdir_p(fppath)
        fnpath = os.path.join(OUTPUT_PATH, 'results/cache/fn')
        if not os.path.exists(fnpath):
            mkdir_p(fnpath)

    #
    groundTruthDict = readGroundTruthValBB()
    number_of_groundtruth_boxes = readTestSet(groundTruthDict)
    boxes_pred_dict, gating_factors = createDictFromPred(DIR_IN)

    # if visualOutput:
    # 	writeGT()
    sorted_boxes_pred, files_with_no_boxes = readAndSortBBs(DIR_IN, groundTruthDict)
    print('numOfGTBBs, {0}'.format(number_of_groundtruth_boxes))
    evaluate(sorted_boxes_pred, groundTruthDict,
             number_of_groundtruth_boxes, threshold=args.iou_threshold)
    if visualOutput:
        draw_boxes(boxes_pred_dict, test_file_sequence, output=OUTPUT_PATH)

    if visualOutput:
        graph_title = 'Results from: {0}'.format(args.boundingboxes)
        graph_output_file = 'consolidated/results_{0}.pdf'.format(int(time.time()))
        plot_gating_values(gating_factors[:, 0], title=graph_title, autosave=False, output_file=graph_output_file)
