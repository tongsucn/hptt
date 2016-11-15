#include <cstdarg>

#include <algorithm>
#include <initializer_list>

#include <hptc/tensor.h>


namespace hptc {

TensorSize::TensorSize()
    : dim_(0),
      size_(nullptr),
      outer_size_(nullptr) {
}


TensorSize::TensorSize(TensorDim dim)
  : dim_(dim),
    size_(new TensorIdx [dim_]) {
}


TensorSize::TensorSize(std::initializer_list<TensorIdx> size)
    : dim_(static_cast<TensorDim>(size.size()),
      size_(new TensorIdx [dim_]) {
  std::copy(size.begin(), size.end(), size_);
}


TensorSize::TensorSize(const TensorSize &size_obj)
    : dim_(size_obj.dim_),
      size_(new TensorIdx [dim_]) {
  this->dim_ = size_obj.dim_;
  std::copy(size_obj.size_, size_obj.size_ + this->dim_, this->size_);
}


TensorSize::TensorSize(TensorSize &&size_obj) noexcept
    : dim_(size_obj.dim_),
      size_(size_obj.size_) {
  size_obj.size_ = nullptr;
}


TensorSize &TensorSize::operator=(const TensorSize &size_obj) {
  this->dim_ = size_obj.dim;

  delete [] this->size_;
  this->size_ = new TensorIdx [this->dim_];
  std::copy(size_obj.size_, size_obj.size_ + this->dim_);

  return *this;
}


TensorSize &TensorSize::operator=(TensorSize &&size_obj) noexcept {
  this->dim_ = size_obj.dim;

  delete [] this->size_;
  this->size_ = size_obj.size_;
  size_obj.size_ = nullptr;

  return *this;
}


TensorSize::~TensorSize() {
  delete [] this->size_;
}


bool TensorSize::operator==(const TensorSize &size_obj) {
  if (this->dim_ != size_obj.dim_)
    return false;

  if (not std::equal(this->size_, this->size_ + this->dim_, size_obj.size_))
    return false;

  return true;
}


inline TensorIdx &TensorSize::operator[](TensorIdx idx) {
  return this->size_[idx];
}


inline TensorDim TensorSize::get_dim() const {
  return this->dim_;
}

}
