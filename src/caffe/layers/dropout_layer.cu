// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {


template <typename Dtype>
__global__ void DropoutForward(const int n, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, Dtype* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    out[index] = in[index] * (mask[index] > threshold);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if ((!mean_on_train_ && Caffe::phase() == Caffe::TRAIN) || !mean_on_test_) {
    CURAND_CHECK(curandGenerate(Caffe::curand_generator(),
        (unsigned int*)(rand_vec_->mutable_gpu_data()), count));
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, (unsigned int*)rand_vec_->gpu_data(), uint_thres_, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    CUDA_CHECK(cudaMemcpy(top_data, bottom_data,
         count * sizeof(Dtype), cudaMemcpyDeviceToDevice));
    caffe_gpu_scal<Dtype>(bottom[0]->count(), Dtype(1. - threshold_), 
			  top_data);
  }
}

template <typename Dtype>
__global__ void DropoutBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold,
    Dtype* out_diff) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    out_diff[index] = in_diff[index] * (mask[index] > threshold);
  }
}

template <typename Dtype>
Dtype DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  CHECK(Caffe::phase() == Caffe::TRAIN);
  if (propagate_down) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const unsigned int* mask = (unsigned int*)rand_vec_->gpu_data();
    const int count = (*bottom)[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, uint_thres_, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
  return Dtype(0);
}

INSTANTIATE_CLASS(DropoutLayer);


}  // namespace caffe
