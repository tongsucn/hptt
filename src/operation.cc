#include <hptc/operation.h>

#include <memory>

#include <hptc/parameter.h>

namespace hptc {

/*
 * Implementation for class Operation
 */
Operation::Operation(const std::shared_ptr<Param> &param,
    Operation *prev = nullptr, Operation *next = nullptr);
    : param(param),
      prev(prev),
      next(next) {
}


inline void Operation::set_prev(Operation *prev) {
  this->prev = prev;
}


inline Operation *Operation::get_prev() {
  return this->prev;
}


inline void Operation::set_next(Operation *next) {
  this->next = next;
}


inline Operation *Operation::get_next() {
  return this->prev;
}


/*
 * Implementation for class OpMicro
 */
OpMicro::OpMicro(const std::shared_ptr<Param> &param)
  : Operation(param) {
}


/*
 * Implementation for class OpMicroCopier
 */
OpMicroCopier::OpMicroCopier(const std::shared_ptr<Param> &param)
  : OpMicro(param) {
}


/*
 * Implementation for class OpMacro
 */
OpMacro::OpMacro(const std::shared_ptr<Param> &param)
  : Operation(param) {
}


/*
 * Implementation for class OpMacroCopier
 */
OpMacroCopier::OpMacroCopier(const std::shared_ptr<Param> &param)
  : OpMacro(param) {
}

}
