#include <cstdarg>

#include <algorithm>
#include <initializer_list>

#include <hptc/tensor.h>


namespace hptc {

TensorSize::TensorSize()
    : dim_(0),
      size_(nullptr) {
}


TensorSize::TensorSize(TensorDim dim)
  : dim_(dim),
    size_(0 != dim ? new TensorIdx [dim] : nullptr) {
}


TensorSize::TensorSize(std::initializer_list<TensorIdx> size)
    : dim_(static_cast<TensorDim>(size.size())),
      size_(0 == this->dim_ ? nullptr : new TensorIdx [dim_]) {
  if (0 != this->dim_)
    std::copy(size.begin(), size.end(), size_);
}


TensorSize::TensorSize(const TensorSize &size_obj)
    : dim_(size_obj.dim_),
      size_(0 == this->dim_ ? nullptr : new TensorIdx [dim_]) {
  if (0 != this->dim_)
    std::copy(size_obj.size_, size_obj.size_ + this->dim_, this->size_);
}


TensorSize::TensorSize(TensorSize &&size_obj) noexcept
    : dim_(size_obj.dim_),
      size_(size_obj.size_) {
  size_obj.size_ = nullptr;
}


TensorSize &TensorSize::operator=(const TensorSize &size_obj) {
  this->dim_ = size_obj.dim_;

  delete [] this->size_;
  this->size_ = 0 == this->dim_ ? nullptr : new TensorIdx [this->dim_];
  if (0 != this->dim_)
    std::copy(size_obj.size_, size_obj.size_ + this->dim_, this->size_);

  return *this;
}


TensorSize &TensorSize::operator=(TensorSize &&size_obj) noexcept {
  this->dim_ = size_obj.dim_;

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

}
