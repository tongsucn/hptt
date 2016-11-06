#include <cstdarg>

#include <algorithm>
#include <initializer_list>

#include <hptc/tensor.h>


namespace hptc {

TensorSize::TensorSize()
    : dim_(0), size_(nullptr), outer_size_(nullptr) {
}


TensorSize::TensorSize(std::initializer_list<TensorIdx> size,
    std::initializer_list<TensorIdx> outer_size = {})
    : dim_(static_cast<TensorDim>(size.size()),
      size_(new TensorIdx [dim_]), outer_size_(new TensorIdx [dim_]) {
  std::copy(size.begin(), size.end(), size_);
  std::copy(outer_size.begin(), outer_size.end(), outer_size_);
}


TensorSize::TensorSize(const TensorSize &size_obj)
    : dim_(size_obj.dim_),
      size_(new TensorIdx [dim_]), outer_size_(new TensorIdx [dim_]) {
  this->dim_ = size_obj.dim_;
  std::copy(size_obj.size_, size_obj.size_ + this->dim_, this->size_);
  std::copy(size_obj.outer_size_, size_obj.outer_size_ + this->dim_,
      this->outer_size_);
}


TensorSize::TensorSize(TensorSize &&size_obj)
    : dim_(size_obj.dim_), size_(size_obj.size_),
      outer_size(size_obj.outer_size_) {
  size_obj.size_ = nullptr;
  size_obj.outer_size_ = nullptr;
}


TensorSize &TensorSize::operator=(const TensorSize &size_obj) {
  this->dim_ = size_obj.dim;

  delete [] this->size_;
  this->size_ = new TensorIdx [this->dim_];
  std::copy(size_obj.size_, size_obj.size_ + this->dim_);

  delete [] this->outer_size_;
  this->outer_size_ = new TensorIdx [this->dim_];
  std::copy(size_obj.outer_size_, size_obj.outer_size_ + this->dim_);

  return *this;
}


TensorSize &TensorSize::operator=(TensorSize &&size_obj) {
  this->dim_ = size_obj.dim;

  delete [] this->size_;
  this->size_ = size_obj.size_;
  size_obj.size_ = nullptr;

  delete [] this->outer_size_;
  this->outer_size_ = size_obj.outer_size_;
  size_obj.outer_size_ = nullptr;

  return *this;
}


bool TensorSize::operator==(const TensorSize &size_obj) {
  if (this->dim_ != size_obj.dim_)
    return false;

  if (not std::equal(this->size_, this->size_ + this->dim_, size_obj.size_))
    return false;

  return true;
}

}
