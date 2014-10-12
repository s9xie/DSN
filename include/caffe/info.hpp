#ifndef CAFFE_NET_INFO_HPP_
#define CAFFE_NET_INFO_HPP_

#include <vector>
#include <iomanip>
#include <limits>
#include <cmath>
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/net.hpp"

using std::setw;
using std::scientific;
using std::left;

namespace caffe {

template <typename Dtype>
class Info {
 public:
  explicit Info(const InfoParameter& info_param,
                shared_ptr<Net<Dtype> > net) : interval_(info_param.interval()), net_(net) {}
  virtual void Iter(Dtype loss, int iter) {
    if (iter % interval_ == 0) {
      print(loss, iter);
    }
  }
  virtual ~Info() {}
 protected:
  virtual void print(Dtype loss, int iter) = 0;
  shared_ptr<Net<Dtype> > net_;
  int interval_;
};

template <typename Dtype>
class WeightRelatedInfo : public Info<Dtype> {
 public:
  explicit WeightRelatedInfo(const InfoParameter& info_param,
                             shared_ptr<Net<Dtype> > net) : Info<Dtype>(info_param, net) {};
  virtual ~WeightRelatedInfo() {};
 protected:
  virtual void print(Dtype loss, int iter) {
    int max_len = 0;
    for (int l = 0; l < this->net_.get()->layers().size(); ++l) {
      Layer<Dtype>& layer = *(this->net_.get()->layers()[l].get());
      if (layer.blobs().size() > 0 && layer.layer_param().name().length() > max_len)
        max_len = layer.layer_param().name().length();
    }
    for (int l = 0; l < this->net_.get()->layers().size(); ++l) {
      Layer<Dtype>& layer = *(this->net_.get()->layers()[l].get());
      for (int b = 0; b < layer.blobs().size(); ++b) {
        Blob<Dtype>& blob = *layer.blobs()[b].get();
        const Dtype* blob_cpu_data = blob.cpu_data();
        const Dtype* blob_cpu_diff = blob.cpu_diff();
        Dtype data_sum = 0;
        Dtype diff_sum = 0;
        Dtype diff_var = 0;
        Dtype temp     = 0;
        for (int i = 0; i < blob.count(); ++i) {
          data_sum += (blob_cpu_data[i] > Dtype(0.)) ? blob_cpu_data[i] : - blob_cpu_data[i];
          diff_sum += (blob_cpu_diff[i] > Dtype(0.)) ? blob_cpu_diff[i] : - blob_cpu_diff[i];
        }
        data_sum /= blob.count();
        diff_sum /= blob.count();
        for (int i = 0; i < blob.count(); ++i) {
          temp = (blob_cpu_diff[i] > Dtype(0.)) ? blob_cpu_diff[i] : - blob_cpu_diff[i];
          diff_var += (temp - diff_sum)*(temp - diff_sum);
        }
        diff_var /= blob.count();
        LOG(INFO) << std::left << std::setw(max_len + 1) << std::setfill(' ')
                  << layer.layer_param().name()
                  << " blob" << b << ": " << std::scientific
                  << data_sum << " [" << diff_sum << "]" << " (" << diff_var << ")";
      }
    }
  }
};

template <typename Dtype>
class BlobRelatedInfo : public Info<Dtype> {
 public:
  explicit BlobRelatedInfo(const InfoParameter& info_param,
                           shared_ptr<Net<Dtype> > net) : Info<Dtype>(info_param, net) {};
  virtual ~BlobRelatedInfo() {}
 protected:
  virtual void print(Dtype loss, int iter) {
    int max_len = 0;
    for (int l = 0; l < this->net_.get()->blob_names().size(); ++l) {
      if (this->net_.get()->blob_names()[l].length() > max_len)
        max_len = this->net_.get()->blob_names()[l].length();
    }
    for (int l = 0; l < this->net_.get()->blobs().size(); ++l) {
      const Blob<Dtype>& blob = *(this->net_.get()->blobs()[l].get());
      const std::string blob_name = this->net_.get()->blob_names()[l];
      const Dtype* blob_cpu_data = blob.cpu_data();
      const Dtype* blob_cpu_diff = blob.cpu_diff();
      Dtype data_max = 0;
      Dtype data_min = 0;
      Dtype diff_max = 0;
      Dtype diff_min = 0;
      Dtype data_mean = 0;
      Dtype diff_mean = 0;
      for (int i = 0; i < blob.count(); ++i) {
        data_mean += blob_cpu_data[i];
        data_max = (blob_cpu_data[i] > data_max) ? blob_cpu_data[i] : data_max;
        data_min = (blob_cpu_data[i] < data_min) ? blob_cpu_data[i] : data_min;
        diff_mean += blob_cpu_diff[i];
        diff_max = (blob_cpu_diff[i] > diff_max) ? blob_cpu_diff[i] : diff_max;
        diff_min = (blob_cpu_diff[i] < diff_min) ? blob_cpu_diff[i] : diff_min;
      }
      data_mean /= blob.count();
      diff_mean /= blob.count();
      LOG(INFO) << std::left << std::setw(max_len + 1) << std::setfill(' ')
                << this->net_.get()->blob_names()[l] << std::scientific
                << " data: (" << data_max << ", " << data_mean << ", " << data_min << ") "
                << "diff: (" << diff_max << ", " << diff_mean << ", " << diff_min << ")";
    }
  }
};

template <typename Dtype>
class MeanTrainLossInfo : public Info<Dtype> {
 public:
  explicit MeanTrainLossInfo(const InfoParameter& info_param,
                             shared_ptr<Net<Dtype> > net) : Info<Dtype>(info_param, net), loss_sum_(0.), iter_num_(0) {}
  virtual void Iter(Dtype loss, int iter) {
    loss_sum_ += loss;
    iter_num_ ++;
    if (iter % this->interval_ == 0) {
      print(loss, iter);
      loss_sum_ = 0.;
      iter_num_ = 0;
    }
  }
  virtual void print(Dtype loss, int iter) {
    LOG(INFO) << "Iter: " << iter << ", loss = " << loss_sum_ / iter_num_;
  }
  virtual ~MeanTrainLossInfo() {}
 protected:
  Dtype loss_sum_;
  int iter_num_;
};

template <typename Dtype>
class LayerLossInfo : public Info<Dtype> {
 public:
  explicit LayerLossInfo(const InfoParameter& info_param,
                             shared_ptr<Net<Dtype> > net) : Info<Dtype>(info_param, net), loss_sum_(0.), iter_num_(0), layer_loss_sum_(0.) {}
  virtual void Iter(Dtype loss, int iter) {
    
	if(layer_loss_info_.empty()) {
 	layer_loss_info_ = this->net_.get()->layer_loss();
   	
        //some sanity check
    	loss_sum_ += loss;
    	layer_loss_sum_ = 0;
        for(int l = 0; l < layer_loss_info_.size(); ++l)
    	    layer_loss_sum_ += layer_loss_info_[l];
    	if(std::fabs(layer_loss_sum_ - loss_sum_) > 1e-1) LOG(FATAL) << "loss sum not equal to sum of all individual loss layers " << layer_loss_sum_ << " vs. " << loss_sum_ << " loss: " << loss << " iter: " << iter;
     	//check over
        }
        else {
        if(layer_loss_info_.size() != this->net_.get()->layer_loss().size()) 
 	    LOG(FATAL) << "layer_loss_info number does not match" << layer_loss_info_.size() << " vs. " << this->net_.get()->layer_loss().size();
    	
	for(int l = 0; l < layer_loss_info_.size(); ++l) 
    	  layer_loss_info_[l] += this->net_.get()->layer_loss()[l];
    	
        //some sanity check
    	loss_sum_ += loss;
        layer_loss_sum_ = 0;
    	for(int l = 0; l < layer_loss_info_.size(); ++l)
    	    layer_loss_sum_ += layer_loss_info_[l];
       if (std::fabs(layer_loss_sum_ - loss_sum_) > 1e-1) //std::numeric_limits<float>::epsilon()) 	
      //if(layer_loss_sum_ != loss_sum_) 
LOG(FATAL) << "loss sum not equal to sum of all individual loss layers(after first iteration)" << layer_loss_sum_ << " vs. " << loss_sum_ << " loss: " << loss << " iter: " << iter;
     	//check over
    
    	
    	iter_num_++;
    	if (iter % this->interval_ == 0) {
    	  print(loss, iter);
    	  loss_sum_ = 0.;
    	  iter_num_ = 0;
    	  layer_loss_info_.clear();
    	  }
  	}
  }
  virtual void print(Dtype loss, int iter) {
    
    for(int l = 0; l < layer_loss_info_.size(); ++l)
       if(layer_loss_info_[l] > 1e-6) 
        LOG(INFO) << "Iter: " << iter << ", loss_layer " << l << ", loss: " << layer_loss_info_[l]/iter_num_;
  }
  virtual ~LayerLossInfo() {}
 protected:
  Dtype loss_sum_;
  Dtype layer_loss_sum_;
  int iter_num_;
  vector<float> layer_loss_info_;
};

template <typename Dtype>
Info<Dtype>* GetInfo(const InfoParameter& info_param, shared_ptr<Net<Dtype> > net) {
  if (info_param.type() == "weight") {
    return new WeightRelatedInfo<Dtype>(info_param, net);
  } else if (info_param.type() == "blob") {
    return new BlobRelatedInfo<Dtype>(info_param, net);
  } else if (info_param.type() == "mean_train_loss") {
    return new MeanTrainLossInfo<Dtype>(info_param, net);
  } else if (info_param.type() == "layer_loss") {
    return new LayerLossInfo<Dtype>(info_param, net);
  } else {
    CHECK(false) << "Unknown info type: " << info_param.type();
  }
  return (Info<Dtype>*)(NULL);
}

}  // namspace caffe

#endif  // CAFFE_NET_INFO_HPP_
