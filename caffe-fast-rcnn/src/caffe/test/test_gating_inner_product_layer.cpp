
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class GatingInnerProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  GatingInnerProductLayerTest()
                            //batch,c, h, w
      : weight_(new Blob<Dtype>(2, 6, 1, 1)),
        bottom_data_(new Blob<Dtype>(2, 3, 1, 1)),
        label_NET1_(new Blob<Dtype>(2, 1, 1, 1)),
        label_NET2_(new Blob<Dtype>(2, 1, 1, 1)),      
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    // FillerParameter filler_param;
    // UniformFiller<Dtype> filler(filler_param);
    // filler.Fill(this->weight_);
    // filler.Fill(this->bottom_data_);
    
    blob_bottom_vec_.push_back(weight_);
    blob_bottom_vec_.push_back(bottom_data_);
    blob_bottom_vec_.push_back(label_NET1_);        
    blob_bottom_vec_.push_back(label_NET2_);        
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~GatingInnerProductLayerTest() { delete bottom_data_; delete weight_; delete blob_top_; 
  delete label_NET1_ ; delete label_NET2_ ;}
  Blob<Dtype>* const weight_;
  Blob<Dtype>* const bottom_data_;
  Blob<Dtype>* const label_NET1_;
  Blob<Dtype>* const label_NET2_;
  Blob<Dtype>* const blob_top_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(GatingInnerProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(GatingInnerProductLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(2);
  shared_ptr<GatingInnerProductLayer<Dtype> > layer(
      new GatingInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2);

  //std::cout<<"this->weight_ "<<this->weight_->shape_string()<<std::endl;
  EXPECT_EQ(this->weight_->num(), 2);
  EXPECT_EQ(this->weight_->height(), 1);
  EXPECT_EQ(this->weight_->width(), 1);
  EXPECT_EQ(this->weight_->channels(), 6);

  EXPECT_EQ(this->bottom_data_->num(), 2);
  EXPECT_EQ(this->bottom_data_->height(), 1);
  EXPECT_EQ(this->bottom_data_->width(), 1);
  EXPECT_EQ(this->bottom_data_->channels(), 3);
}

TYPED_TEST(GatingInnerProductLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(2);
    shared_ptr<GatingInnerProductLayer<Dtype> > layer(
        new GatingInnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_); 

    Dtype* weight_data = this->blob_bottom_vec_[0]->mutable_cpu_data();

    weight_data[0] = 0.25;
    weight_data[1] = 0.75;
    weight_data[2] = 0.6;
    weight_data[3] = 0.4;
    weight_data[4] = 0.8;
    weight_data[5] = 0.2;

    weight_data[6] = 0.7;
    weight_data[7] = 0.3;
    weight_data[8] = 0.1;
    weight_data[9] = 0.9;
    weight_data[10] = 0.45;
    weight_data[11] = 0.55;


    Dtype* bottom_data_data = this->blob_bottom_vec_[1]->mutable_cpu_data();
    bottom_data_data[0] = 0.55;
    bottom_data_data[1] = 0.35;
    bottom_data_data[2] = 0.10;
    bottom_data_data[3] = 0.6;
    bottom_data_data[4] = 0.3;
    bottom_data_data[5] = 0.1;



    Dtype* label_NET1_data = this->blob_bottom_vec_[2]->mutable_cpu_data();
    label_NET1_data[0] = 0;
    label_NET1_data[1] = 1; 
    Dtype* label_NET2_data = this->blob_bottom_vec_[3]->mutable_cpu_data();
    label_NET2_data[0] = 0;
    label_NET2_data[1] = 1;  

    Blob<Dtype>*  result(new Blob<Dtype>());
    vector<int> result_shape(2);
    result_shape[0] = 2;
    result_shape[1] = 2;
    result->Reshape(result_shape);
    Dtype* result_data = result->mutable_cpu_data();

    result_data[0] = 0.42750;
    result_data[1] = 0.57250;
    result_data[2] = 0.49500;
    result_data[3] = 0.50500;

    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();
    EXPECT_NEAR(data[0], result_data[0],1.e-3);
    EXPECT_NEAR(data[1], result_data[1],1.e-3);
    EXPECT_NEAR(data[2], result_data[2],1.e-3);
    EXPECT_NEAR(data[3], result_data[3],1.e-3);
/*    for (int i = 0; i < result->count(); ++i) {
      //EXPECT_GE(data[i], 1.);
      std::cout<<"data["<<i<<"]"<<data[i]<<std::endl;
    }*/
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(GatingInnerProductLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(2);
    inner_product_param->mutable_weight_filler()->set_type("gaussian");
    inner_product_param->mutable_bias_filler()->set_type("gaussian");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);
    shared_ptr<GatingInnerProductLayer<Dtype> > layer(
        new GatingInnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_); 

    Dtype* weight_data = this->blob_bottom_vec_[0]->mutable_cpu_data();
    weight_data[0] = 0.25;
    weight_data[1] = 0.75;
    weight_data[2] = 0.6;
    weight_data[3] = 0.4;
    weight_data[4] = 0.8;
    weight_data[5] = 0.2;

    weight_data[6] = 0.7;
    weight_data[7] = 0.3;
    weight_data[8] = 0.1;
    weight_data[9] = 0.9;
    weight_data[10] = 0.45;
    weight_data[11] = 0.55;

    Dtype* bottom_data_data = this->blob_bottom_vec_[1]->mutable_cpu_data();
    bottom_data_data[0] = 0.55;
    bottom_data_data[1] = 0.35;
    bottom_data_data[2] = 0.10;
    bottom_data_data[3] = 0.6;
    bottom_data_data[4] = 0.3;
    bottom_data_data[5] = 0.1;

    Dtype* label_NET1 = this->blob_bottom_vec_[2]->mutable_cpu_data();
    label_NET1[0] = 1;
    label_NET1[1] = 0;
    Dtype* label_NET2 = this->blob_bottom_vec_[3]->mutable_cpu_data();  
    label_NET2[0] = 1;
    label_NET2[1] = 0;

    Dtype* blob_top_vec_data = this->blob_top_vec_[0]->mutable_cpu_diff();
    blob_top_vec_data[0] = 0.3125;
    blob_top_vec_data[1] = 0.6875;
    blob_top_vec_data[2] = 0.79;
    blob_top_vec_data[3] = 0.21;
    const int check_bottom = -1;
    vector<bool> propagate_down(this->blob_bottom_vec_.size(), check_bottom < 0);
    layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

    Dtype* bottom_backward = this->blob_bottom_vec_[1]->mutable_cpu_diff();
    for(int i = 0; i < this->blob_bottom_vec_[1]->count() ; ++i){
      std::cout << "GatingInnerProductLayer Backward bottom_backward diff i" << i << " : " << bottom_backward[i] << std::endl;
    }

    Blob<Dtype>*  result(new Blob<Dtype>());
    vector<int> result_shape(2);
    result_shape[0] = 2;
    result_shape[1] = 3;
    result->Reshape(result_shape);
    Dtype* result_data = result->mutable_cpu_data();

    result_data[0] = 0.59375;
    result_data[1] = 0.46250;
    result_data[2] = 0.38750;
    result_data[3] = 0.61600;
    result_data[4] = 0.26800;
    result_data[5] = 0.47100;


    EXPECT_NEAR(bottom_backward[0], result_data[0],1.e-3);
    EXPECT_NEAR(bottom_backward[1], result_data[1],1.e-3);
    EXPECT_NEAR(bottom_backward[2], result_data[2],1.e-3);
    EXPECT_NEAR(bottom_backward[3], result_data[3],1.e-3);
    EXPECT_NEAR(bottom_backward[4], result_data[4],1.e-3);
    EXPECT_NEAR(bottom_backward[5], result_data[5],1.e-3);
    std::cout << "Passed backward test" << std::endl;

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe







