#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <chrono>

namespace caffe {



template <typename Dtype>
void GatingInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Dtype* weight = bottom[0]->mutable_cpu_data();
  vector<int> weight_reshaped_shape(2);
  weight_reshaped_shape[0] = K_;
  weight_reshaped_shape[1] = M_*N_;
  Blob<Dtype>*  weight_reshaped(new Blob<Dtype>());
  weight_reshaped->Reshape(weight_reshaped_shape);
  Dtype* weight_reshaped_data = weight_reshaped->mutable_cpu_data();
  int it = 0;
  for (int c = 0; c < M_ ; ++c) {
    for (int j = 0; j < K_; ++j) {
      for (int i = 0; i < N_; ++i) { 
        weight_reshaped_data[ j*M_*N_ + c*N_ + i] = weight[it];
/*        std::cout << "at" << j*M_*N_ + c*N_ + i << std::endl; 
        std::cout << "Weight raw: " << "i"  << it  << " " << weight[it] << std::endl;*/
        it += 1;        
      }
    }
  }
  //const_cast<Dtype*>(weight);
  //const_cast<Dtype*>(weight_reshaped_data);
  const Dtype* top_diff = top[0]->cpu_diff();
  vector<int> top_diff_reshaped_shape(2);
  top_diff_reshaped_shape[0] = M_;
  top_diff_reshaped_shape[1] = M_*N_;
  Blob<Dtype>*  top_diff_reshaped(new Blob<Dtype>());
  top_diff_reshaped->Reshape(top_diff_reshaped_shape);
  Dtype* top_diff_reshaped_data = top_diff_reshaped->mutable_cpu_diff();
  // Reshape top_diff to diagonal matrix
  int idx2 = 0;
  int aux = 1;
  int m = 0;    
    for(int i = 0; i < M_*N_; ++i ) {     
      for (int j = 0; j < M_; ++j) {
        int it = i*M_ + j;
        //m = c*K_;
        if(it == idx2){
            //std::cout << "m: " << m << " top_diff_reshaped_data[m]: " << top_diff[m] << " it "  << it << " idx2 "  << idx2 << std::endl;          
            top_diff_reshaped_data[it] = top_diff[m];
            if (aux<N_){
              idx2 +=1;
              aux++;
            } else if (aux==N_){
              idx2=idx2+M_*N_+1;
              aux=1;
            }
            m += 1;            
        }
        else{
          top_diff_reshaped_data[it] = 0;  
        }
       //std::cout << "i: " << i << "GatingInnerProductLayer Backward top_diff_reshaped_data: it" << it << " data: " <<  top_diff_reshaped_data[it] << std::endl; 
      }
    }
  //const_cast<Dtype*>(top_diff_reshaped_data);
  // Gradient with respect to bottom data
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, M_*N_, (Dtype)1.,
      top_diff_reshaped_data, weight_reshaped_data, (Dtype)0.,
      bottom[1]->mutable_cpu_diff());
  delete top_diff_reshaped_data;
  delete weight_reshaped_data;
}

template <typename Dtype>
void GatingInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {  
  // Contains softmax outputs of individual experts
  Dtype* weight = bottom[0]->mutable_cpu_data();
  // Contains concatenated feature represenation of experts 
  Dtype* bottom_data = bottom[1]->mutable_cpu_data();
  //Dtype* bottom_data = bottom[1]->mutable_gpu_data();
  const Dtype* label_NET1 = bottom[2]->cpu_data();
  const Dtype* label_NET2 = bottom[3]->cpu_data();  
  Dtype* top_data = top[0]->mutable_gpu_data();
  vector<int> weight_shape = bottom[0]->shape();
  vector<int> bottom_data_shape = bottom[1]->shape();

  //const int weight_count = bottom[0]->count();
  //const int bottom_data_count = bottom[1]->count();
//   //This needs to be of Size M_ * N_ * K_
//   //std::cout << " GatingInnerProductLayer weight_shape: " << weight_shape[0] << " "  << weight_shape[1] <<std::endl;
//   //This needs to be of Size M_ K_
// /*  std::cout << " GatingInnerProductLayer bottom_data_shape: " << bottom_data_shape[0] << " " << bottom_data_shape[1] << std::endl;
//   std::cout << " weight_count: " << weight_count << std::endl;     
//   std::cout << " bottom_data_count: " << bottom_data_count << std::endl;*/
//   // Check if labels from data layers for both expert networks are consistent
  for (int li = 0 ; li <  bottom[2]->count() ; ++li) {
    const int label_value_NET1 = static_cast<int>(label_NET1[li]);
    const int label_value_NET2 = static_cast<int>(label_NET2[li]);
    //std::cout << "label_value_NET1 " << label_value_NET1 << " label_value_NET2 " << label_value_NET2 << std::endl;   
    if( label_value_NET1 != label_value_NET2 ) {
      LOG(FATAL) << "Groundtruth of labels for expert networks are not consistent (" << label_value_NET1 << " != "  << label_value_NET2 << " ) something nasty is happening, check the data layers";
    }
  }
  for(int i = 0; i < bottom[1]->count(); ++i ) {
    if(std::isnan(bottom_data[i])){
      std::cout << "Bottom data forward is nan!!! : i" << i << " " << bottom_data[i] << std::endl;
      exit(1);
    }
  }

//   // For testing purposes

  bool Debug_mode = false;
  if(Debug_mode){
    bottom_data[0] = 0.5;
    bottom_data[1] = 0.5;
    //bottom_data[2] = 0.33;
    std::cout << "Running gating layer in debug mode" << std::endl;    
  }
  


  vector<int> mirrored_bottom_data_shape(2);
  mirrored_bottom_data_shape[0] = M_;
  mirrored_bottom_data_shape[1] = M_*K_;
  //Blob<Dtype>*  mirrored_bottom_data(new Blob<Dtype>());
  boost::shared_ptr<Blob<Dtype> > mirrored_bottom_data(new Blob<Dtype>()); 
  mirrored_bottom_data->Reshape(mirrored_bottom_data_shape);
  Dtype* mirrored_bottom_data_data = mirrored_bottom_data->mutable_cpu_data();
  // Reshape bottom data to diagonal matrix
  int idx2 = 0;
  int aux = 1;
  int m = 0;
  //std::cout<<"M_: "<<M_<<" K_: "<<K_<<" N_: "<<N_<<std::endl;
  //auto  t1 = std::chrono::high_resolution_clock::now();
  #pragma omp for
  for(int i = 0; i < M_*K_; ++i ) {     
    for (int j = 0; j < M_; ++j) {
      int it = i*M_ + j;
      if(it == idx2){
          if(std::isnan(bottom_data[m])){
            std::cout << "bottom_data[" << m << "]: " << bottom_data[m] << std::endl; 
            exit(1);
          }
          if(Debug_mode){
            mirrored_bottom_data_data[it] = bottom_data[aux-1];
          }
          else{
            mirrored_bottom_data_data[it] = bottom_data[m];    
          }
          if (aux<K_){
            idx2 +=1;
            aux++;
          } else if (aux==K_){
            idx2=idx2+M_*K_+1;
            aux=1;
          }
          m += 1;            
      }
      else{
        mirrored_bottom_data_data[it] = 0;  
      }
      //std::cout << "i: " << i << " mirrored_bottom_data: it" << it << " data: " <<  mirrored_bottom_data_data[it] << std::endl; 
    }
  }
// auto t2 = std::chrono::high_resolution_clock::now();
//   typedef std::chrono::duration<float> fsec;
//   fsec fs = t2 - t1;
//   std::cout << "First loop took " << fs.count() << " s\n";



//   //const_cast<Dtype*>(mirrored_bottom_data_data);
  vector<int> weight_reshaped_shape(2);
  weight_reshaped_shape[0] = N_;
  weight_reshaped_shape[1] = M_*K_;
  //Blob<Dtype>*  weight_reshaped(new Blob<Dtype>());
  boost::shared_ptr<Blob<Dtype> > weight_reshaped(new Blob<Dtype>()); 
  weight_reshaped->Reshape(weight_reshaped_shape);
  Dtype* weight_reshaped_data = weight_reshaped->mutable_cpu_data();
  //t1 = std::chrono::high_resolution_clock::now();
  int it = 0;
  for (int c = 0; c < M_ ; ++c) {
    for (int j = 0; j < K_; ++j) {
      for (int i = 0; i < N_; ++i) { 
        weight_reshaped_data[ i*M_*K_ + c*K_ + j] = weight[it];
        //std::cout << "Weight raw: " << "i"  << it  << " " << weight[it] << std::endl;
        it += 1;        
      }
    }
  }

   caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, M_*K_, (Dtype)1.,
       mirrored_bottom_data->mutable_gpu_data(), weight_reshaped->mutable_gpu_data(), (Dtype)0., top_data);


  bool writeToFile = true;
  if (writeToFile){

    std::ofstream probout;  
    if (this->phase_ == TRAIN) {
      probout.open("/tmp/gating_probout_train.txt",std::ofstream::out | std::ofstream::app);  
    }
    else if (this->phase_ == TEST) {
      probout.open("/tmp/gating_probout_test.txt",std::ofstream::out | std::ofstream::app);  
    }
    else{
      LOG(FATAL) << "Couldn't open file /tmp/gating_probout because phase_ was not defined";
    }
      // Write softmax outputs in trainlist
      for (int c = 0; c < M_ ; ++c) {
      for (int j = 0; j < K_ ; ++j) {
        probout << "expert[" << j << "]{";
        for (int i = 0; i < N_ ; ++i) {
          int index = ( c * K_ + j  ) * N_ + i;
          probout << weight[index] << " ";
        }
        probout << "}:";
      }
      probout << "gating assignment{" ;
        for (int j = 0; j < K_ ; ++j) {
        int index = c*K_ + j;
       probout << bottom_data[index] << " ";
      }
      probout << "}:";
      probout << "gate_out{";
      for (int gi = 0; gi < N_ ; ++gi) {
          int tdi = c*N_ + gi; 
          probout << top[0]->mutable_cpu_data()[tdi] << " ";
      }
      probout << "}:";
      const int label_value = static_cast<int>(label_NET1[c]);
      probout << "label[" << label_value << "]" << std::endl;   
     }  
      probout.close();
  }


}





INSTANTIATE_LAYER_GPU_FUNCS(GatingInnerProductLayer);

}  // namespace caffe
