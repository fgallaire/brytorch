// wasthon_census.cpp — read-only C-side introspection for the bridge's
// demoted-instance census (the bilateral death test): what kind of object a
// candidate is, whether the C++ side still holds its payload (TensorImpl
// use count), how many bytes freeing it would recover, and allocator state.
// Called from JS (wasthon.js reclaim dry-run); never mutates anything.
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include <malloc.h>

extern "C" {

// 1 = torch tensor (THPVariable), 2 = type object, 3 = pybind11 instance,
// 0 = other C instance.
int wasthon_census_kind(PyObject* o) {
  if (THPVariable_Check(o)) return 1;
  if (PyType_Check(o)) return 2;
  if (pybind11::detail::get_type_info(Py_TYPE(o))) return 3;
  return 0;
}

// what=0: TensorImpl use_count (1 = only this wrapper's cdata holds it —
// the C++ side agrees the tensor dies with the wrapper); 1: weak use_count;
// 2: nbytes of the underlying storage. -1 on a non-tensor.
double wasthon_census_tensor(PyObject* o, int what) {
  if (!THPVariable_Check(o)) return -1;
  const auto& v = THPVariable_Unpack(o);
  if (!v.defined()) return what == 2 ? 0 : 1;
  switch (what) {
    case 0: return (double)v.use_count();
    case 1: return (double)v.weak_use_count();
    case 2: return (double)v.nbytes();
    // storage sharing: freeing the tensor returns its bytes only when the
    // StorageImpl's count drops to zero — a shared storage stays pinned.
    case 3: return v.has_storage()
        ? (double)v.storage().use_count() : 0;
    // storage identity (StorageImpl address) — dedup key for "who owns the
    // heap": many tensors, one storage, count the bytes once.
    case 4: return v.has_storage()
        ? (double)(intptr_t)v.storage().unsafeGetStorageImpl() : 0;
  }
  return -1;
}

// The PyObjects a tensor's C++ state holds that NO Python-level walk can see.
// `t.grad` is an accessor over autograd metadata, not a stored attribute, so
// scanning t's __dict__ finds nothing while the C++ genuinely holds the other
// tensor — which is precisely what test_tensor_cycle_via_dict asserts with
// `z.grad = x`. tp_traverse is not an answer here: it deliberately does NOT
// report grad (see NOTE [ PyObject Traversal ] in python_variable.cpp), because
// CPython gets that liveness from the refcount instead, which we do not have.
// So the bridge asks here, for REACHABILITY ONLY. `i` selects a fixed slot
// (0 = grad, 1 = grad_fn); null just means that slot is empty, so the caller
// walks the whole small range. Read-only, borrows, mutates nothing.
PyObject* wasthon_census_edge(PyObject* o, int i) {
  if (!THPVariable_Check(o)) return nullptr;
  const auto& v = THPVariable_Unpack(o);
  if (!v.defined()) return nullptr;
  auto* meta = torch::autograd::impl::get_autograd_meta(v);
  if (!meta) return nullptr;
  switch (i) {
    case 0: {  // .grad
      const auto& g = meta->grad_;
      if (g.defined()) {
        return g.unsafeGetTensorImpl()->pyobj_slot()->load_pyobj();
      }
      return nullptr;
    }
    case 1: {  // .grad_fn (the graph node's wrapper, when it has one)
      const auto& gf = meta->grad_fn_;
      if (gf) return gf->pyobj_slot()->load_pyobj();
      return nullptr;
    }
  }
  return nullptr;
}

// what=0: total free bytes in the allocator (fordblks); 1: allocated bytes
// (uordblks). The heap high-water is fordblks+uordblks (+ overhead).
double wasthon_census_mallinfo(int what) {
  struct mallinfo mi = mallinfo();
  return what == 0 ? (double)mi.fordblks : (double)mi.uordblks;
}

}  // extern "C"
