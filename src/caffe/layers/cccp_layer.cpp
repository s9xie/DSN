// Copyright 2014 Min Lin

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CCCPPoolingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "CCCP Pooling Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "CCCP Pooling Layer takes a single blob as output.";

  NUM_OUTPUT_ = this->layer_param_.num_output();
  GROUP_      = this->layer_param_.group();
  biasterm_   = this->layer_param_.biasterm();

  // Figure out the dimensions
  CHANNEL_ = bottom[0]->channels();
  REST_ = bottom[0]->height() * bottom[0]->width();
  NUM_ = bottom[0]->num();

  CHECK_EQ(CHANNEL_%GROUP_, 0) << "CCCP Pooling input channel number is not divisible by group number.";

  (*top)[0]->Reshape(bottom[0]->num(), GROUP_*NUM_OUTPUT_, bottom[0]->height(), bottom[0]->width());
  
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (biasterm_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, GROUP_*NUM_OUTPUT_, CHANNEL_/GROUP_));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(
        GetFiller<Dtype>(this->layer_param_.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (biasterm_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, GROUP_*NUM_OUTPUT_));
      shared_ptr<Filler<Dtype> > bias_filler(
          GetFiller<Dtype>(this->layer_param_.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  // Setting up the bias multiplier
  if (biasterm_) {
    bias_multiplier_.reset(new SyncedMemory(REST_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < REST_; ++i) {
        bias_multiplier_data[i] = 1.;
    }
  }
}

template <typename Dtype>
void CCCPPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const int weight_offset = NUM_OUTPUT_*CHANNEL_/GROUP_;
  const int bottom_group_offset = REST_*CHANNEL_/GROUP_;
  const int top_group_offset = REST_*NUM_OUTPUT_;

  for (int n = 0; n < NUM_; ++n) {
    for (int g = 0; g < GROUP_; ++g) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, NUM_OUTPUT_, REST_, CHANNEL_/GROUP_, (Dtype)1.,
      weight + g*weight_offset, 
      bottom_data + bottom[0]->offset(n)+g*bottom_group_offset, 
      (Dtype)0., 
      top_data + (*top)[0]->offset(n)+g*top_group_offset);
    }
    if (biasterm_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, GROUP_*NUM_OUTPUT_, REST_, 1, (Dtype)1.,
      this->blobs_[1]->cpu_data(), 
      reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), (Dtype)1.,
      top_data + (*top)[0]->offset(n));
    }
  }
}

template <typename Dtype>
void CCCPPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int weight_offset = NUM_OUTPUT_*CHANNEL_/GROUP_;
  const int bottom_group_offset = REST_*CHANNEL_/GROUP_;
  const int top_group_offset = REST_*NUM_OUTPUT_;

  for (int n = 0; n < NUM_; ++n) {
    for (int g = 0; g < GROUP_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, NUM_OUTPUT_, REST_, CHANNEL_/GROUP_, (Dtype)1.,
      weight + g*weight_offset, 
      bottom_data + bottom[0]->offset(n)+g*bottom_group_offset, 
      (Dtype)0., 
      top_data + (*top)[0]->offset(n)+g*top_group_offset);
    }
    if (biasterm_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, GROUP_*NUM_OUTPUT_, REST_, 1, (Dtype)1.,
      this->blobs_[1]->gpu_data(), 
      reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()), (Dtype)1.,
      top_data + (*top)[0]->offset(n));
    }
  }
}

template <typename Dtype>
Dtype CCCPPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* bias_diff = NULL;
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();

  const int weight_offset = NUM_OUTPUT_*CHANNEL_/GROUP_;
  const int bottom_group_offset = REST_*CHANNEL_/GROUP_;
  const int top_group_offset = REST_*NUM_OUTPUT_;

  // Gradient w.r.t. bias
  if (biasterm_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    memset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count());
    for (int n = 0; n < NUM_; ++n) {
      caffe_cpu_gemv<Dtype>(CblasNoTrans, NUM_OUTPUT_*GROUP_, REST_, (Dtype)1., 
        top_diff + top[0]->offset(n),
        reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
        (Dtype)1.,
        bias_diff);
    }
  }

  memset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count());
  for (int n = 0; n < NUM_; ++n) {
    // The gradient will be accumulated
    for (int g = 0; g < GROUP_; ++g) {
      // Gradient with respect to weight
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, NUM_OUTPUT_, CHANNEL_/GROUP_, REST_, (Dtype)1.,
        top_diff + top[0]->offset(n) + g*top_group_offset, 
        bottom_data + (*bottom)[0]->offset(n) + g*bottom_group_offset, 
        (Dtype)1.,
        weight_diff + g*weight_offset);

      // Gradient w.r.t. bottom data if necessary
      if (propagate_down) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, CHANNEL_/GROUP_, REST_, NUM_OUTPUT_, (Dtype)1.,
            weight + g*weight_offset,
            top_diff + top[0]->offset(n) + g*top_group_offset, 
            (Dtype)0.,
            bottom_diff + (*bottom)[0]->offset(n) + g*bottom_group_offset);
      }
    }
  }
  return Dtype(0);
}

template <typename Dtype>
Dtype CCCPPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* bias_diff = NULL;
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();

  const int weight_offset = NUM_OUTPUT_*CHANNEL_/GROUP_;
  const int bottom_group_offset = REST_*CHANNEL_/GROUP_;
  const int top_group_offset = REST_*NUM_OUTPUT_;

  // Gradient w.r.t. bias
  if (biasterm_) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    CUDA_CHECK(cudaMemset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count()));
    for (int n = 0; n < NUM_; ++n) {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, NUM_OUTPUT_*GROUP_, REST_, (Dtype)1., 
	top_diff + top[0]->offset(n),
        reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
        (Dtype)1.,
        bias_diff);
    }
  }
  
  CUDA_CHECK(cudaMemset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count()));
  for (int n = 0; n < NUM_; ++n) {
    // The gradient will be accumulated
    for (int g = 0; g < GROUP_; ++g) {
      // Gradient with respect to weight
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, NUM_OUTPUT_, CHANNEL_/GROUP_, REST_, (Dtype)1.,
        top_diff + top[0]->offset(n) + g*top_group_offset, 
        bottom_data + (*bottom)[0]->offset(n) + g*bottom_group_offset, 
        (Dtype)1.,
        weight_diff + g*weight_offset);
      // Gradient w.r.t. bottom data if necessary
      if (propagate_down) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, CHANNEL_/GROUP_, REST_, NUM_OUTPUT_, (Dtype)1.,
            weight + g*weight_offset,
            top_diff + top[0]->offset(n) + g*top_group_offset, 
            (Dtype)0.,
            bottom_diff + (*bottom)[0]->offset(n) + g*bottom_group_offset);
      }
    }
  }
  return Dtype(0);
}

INSTANTIATE_CLASS(CCCPPoolingLayer);

}
