// torch_stubs.cpp — C++ stubs for the v1-excluded torch subsystems
// (dynamo eval-frame/guards, profiler captured-traceback, onnx export,
// libshm): their sources don't compile against wasthon (CPython-internal
// frames / ungenerated protobuf) or target multi-process shared memory,
// but linked TUs reference these entry points. Python-side, the matching
// modules are stubbed raise-at-use in the VFS.
#include <torch/csrc/dynamo/guards.h>
#include <torch/csrc/profiler/python/combined_traceback.h>
#include <libshm.h>
#include <string>
#include <memory>

namespace torch::autograd::profiler::python_tracer {
void init() {}
}  // namespace torch::autograd::profiler::python_tracer

namespace torch::dynamo {
void initDynamoBindings(PyObject* torch) { (void)torch; }
bool get_is_in_mode_without_ignore_compile_internals() { return false; }
TensorCheck::TensorCheck(
    const LocalState& state,
    PyTypeObject* pt,
    c10::DispatchKeySet dispatch_key_set,
    at::ScalarType dtype,
    at::DeviceIndex device_index,
    bool requires_grad,
    std::vector<std::optional<c10::SymInt>> dynamic_dims_sizes,
    std::vector<std::optional<c10::SymInt>> dynamic_dims_strides)
    : pytype(pt),
      dispatch_key_(state.apply(dispatch_key_set).raw_repr()),
      dtype_(dtype),
      device_index_(device_index),
      requires_grad_(requires_grad),
      sizes_(std::move(dynamic_dims_sizes)),
      strides_(std::move(dynamic_dims_strides)),
      dim_(static_cast<int64_t>(sizes_.size())) {}
bool TensorCheck::check(
    const LocalState& state,
    const c10::DispatchKeySet& dispatch_key_set,
    const at::ScalarType& dtype,
    const c10::Device& device,
    const c10::SymIntArrayRef& dynamic_dims_sizes,
    const c10::SymIntArrayRef& dynamic_dims_strides,
    const bool& requires_grad) {
  (void)state; (void)dispatch_key_set; (void)dtype; (void)device;
  (void)dynamic_dims_sizes; (void)dynamic_dims_strides; (void)requires_grad;
  return false;
}
}  // namespace torch::dynamo

namespace torch {
void freeDeadCapturedTracebackFrames() {}
void installCapturedTracebackPython() {}
std::vector<pybind11::object> py_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize) {
  (void)to_symbolize;
  return {};
}
}  // namespace torch

// --- libshm: single-process wasm has no shared-memory manager ---
void libshm_init(const char* manager_exec_path) { (void)manager_exec_path; }
THManagedMapAllocatorInit::THManagedMapAllocatorInit(
    const char* manager_handle, const char* filename)
    : manager_handle_(manager_handle ? manager_handle : "") {
  (void)filename;
}
at::DataPtr THManagedMapAllocator::makeDataPtr(
    const char* manager_handle, const char* filename, int flags, size_t size) {
  (void)manager_handle; (void)filename; (void)flags; (void)size;
  return at::DataPtr();
}
THManagedMapAllocator* THManagedMapAllocator::fromDataPtr(const at::DataPtr& dptr) {
  (void)dptr;
  return nullptr;
}
