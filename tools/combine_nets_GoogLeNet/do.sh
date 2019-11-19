#!/usr/bin/env bash
#CAFFEBIN="/sysgen/code/ais_fast_rcnn_MoDE/caffe-fast-rcnn"
CAFFEBIN="/sysgen/code/ais_caffe"
DEPLOYOUT="deploy-merged.prototxt"
NUM_MODELS=2
for i in `seq 0 0`;
do	
	datasetdir_left=$PWD/left/
	datasetdir_right=$PWD/right/

	left_weight=`printf "%ss%02d_left.caffemodel" "${datasetdir_left}" "${i}"`
	right_weight=`printf "%ss%02d_right.caffemodel" "${datasetdir_right}" "${i}"`

	out_weight=googLeNet-fus$NUM_MODELS.caffemodel
	echo $out_weight
	$CAFFEBIN/build/tools/combine_networks.bin $DEPLOYOUT $left_weight $right_weight $out_weight
done    
    	 


