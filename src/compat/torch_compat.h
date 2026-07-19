/* torch_compat.h — C-API surface torch/csrc needs beyond wasthon.h.
 * Passed with -include ahead of every torch/csrc TU (same pattern as
 * numbry's sp_compat.h). Declarations compile-side; implementations land
 * in torch_compat.c / the bridge at link time. Kept out of wasthon.h so
 * the torch port doesn't touch the wasthon4 working tree.
 *
 * Known semantic debts (compile now, resolve at run stage):
 *  - _wasthon_code / PyTracebackObject member layout: only pybind11's
 *    error-formatting path dereferences them; bridge must materialize or
 *    the path must stay cold.
 *  - Py_TPFLAGS_MANAGED_DICT: defined with CPython's bit; the bridge
 *    ignores it (no managed dicts), Clear/Visit are stubs.
 */
#ifndef WASTHON_TORCH_COMPAT_H
#define WASTHON_TORCH_COMPAT_H
#include <Python.h>
#include <unistd.h>   /* getpid/_exit: CPython's Python.h pulls it in */


#ifdef __cplusplus
extern "C" {
#endif

/* CPython-internal eval frame — opaque; only dynamo/profiler take pointers */
typedef struct _PyInterpreterFrame _PyInterpreterFrame;

/* thread-specific storage: single-thread stubs at link */
typedef int Py_tss_t;
#define Py_tss_NEEDS_INIT 0
Py_tss_t *PyThread_tss_alloc(void);
int   PyThread_tss_create(Py_tss_t *key);
void  PyThread_tss_delete(Py_tss_t *key);
int   PyThread_tss_set(Py_tss_t *key, void *value);
void *PyThread_tss_get(Py_tss_t *key);

int _PyEval_SliceIndex(PyObject *v, Py_ssize_t *pi);

/* CPython's GET macros cast their argument and yield an lvalue (array
 * element); wasthon's map to typed functions. Restore the cast, and in
 * C++ return a reference to a scratch slot: torch binds `auto&` to it
 * (Generator pickleSetState) and reads it back immediately. */
#undef PyTuple_GET_ITEM
#ifdef __cplusplus
static inline PyObject *&_wasthon_tuple_get_lv(PyObject *t, Py_ssize_t i) {
    static PyObject *slot;
    slot = PyTuple_GetItem(t, i);
    return slot;
}
#define PyTuple_GET_ITEM(tup, i) _wasthon_tuple_get_lv((PyObject *)(tup), (i))
#else
#define PyTuple_GET_ITEM(tup, i) PyTuple_GetItem((PyObject *)(tup), (i))
#endif
#undef PyTuple_GET_SIZE
#define PyTuple_GET_SIZE(tup) PyTuple_Size((PyObject *)(tup))

/* --- structs torch/pybind11 dereference directly --- */
/* CPython layout (header + 3 slots). The bridge materializes a backing
 * struct for every slice handle (wrap() in wasthon.js): ob_refcnt @0,
 * start/stop/step handles @4/8/12 — so ->start/stop/step reads resolve. */
typedef struct {
    PyObject_HEAD
    PyObject *start;
    PyObject *stop;
    PyObject *step;
} PySliceObject;

struct _wasthon_code {          /* completes wasthon.h's forward decl */
    PyObject *co_filename;
    PyObject *co_name;
    int co_argcount;
    int co_firstlineno;
    int co_nfreevars;
    int co_ncellvars;
};

typedef struct _traceback {
    struct _traceback *tb_next;
    PyFrameObject *tb_frame;
    int tb_lasti;
    int tb_lineno;
} PyTracebackObject;

typedef PyObject *(*PyCFunctionWithKeywords)(PyObject *, PyObject *, PyObject *);

/* --- type checks / accessors --- */
int PyWeakref_Check(PyObject *o);
int PyFrozenSet_Check(PyObject *o);
int PyGen_Check(PyObject *o);
int PyCode_Check(PyObject *o);
int PyMethod_Check(PyObject *o);
int PyInstanceMethod_Check(PyObject *o);
PyObject *PyMethod_GET_FUNCTION(PyObject *m);
PyObject *PyInstanceMethod_GET_FUNCTION(PyObject *m);
PyObject *PyInstanceMethod_New(PyObject *func);
PyObject *PyStaticMethod_New(PyObject *func);
extern PyTypeObject PyStaticMethod_Type;
PyObject *PyType_GetName(PyTypeObject *type);
static inline int PyType_HasFeature(PyTypeObject *t, unsigned long f) {
    return (t->tp_flags & f) != 0;
}

/* --- objects / attributes --- */
int PyObject_HasAttr(PyObject *o, PyObject *name);
int PyObject_DelAttr(PyObject *o, PyObject *name);
int PyObject_DelAttrString(PyObject *o, const char *name);
int PyObject_GetOptionalAttrString(PyObject *o, const char *name, PyObject **out);
PyObject *PyDict_SetDefault(PyObject *d, PyObject *key, PyObject *dflt);
Py_ssize_t PySet_Size(PyObject *s);
int PySet_Clear(PyObject *s);
PyObject *PyByteArray_FromObject(PyObject *o);
PyObject *PyMemoryView_FromBuffer(const Py_buffer *view);
void PyUnicode_InternInPlace(PyObject **p);
#define PyUnicode_CHECK_INTERNED(op) 0
int PyCapsule_SetDestructor(PyObject *capsule, PyCapsule_Destructor destructor);

/* --- in-place number protocol (bridge has the nb_inplace slots) --- */
PyObject *PyNumber_InPlaceAdd(PyObject *a, PyObject *b);
PyObject *PyNumber_InPlaceSubtract(PyObject *a, PyObject *b);
PyObject *PyNumber_InPlaceMultiply(PyObject *a, PyObject *b);
PyObject *PyNumber_InPlaceTrueDivide(PyObject *a, PyObject *b);
PyObject *PyNumber_InPlaceLshift(PyObject *a, PyObject *b);
PyObject *PyNumber_InPlaceRshift(PyObject *a, PyObject *b);
PyObject *PyNumber_InPlaceAnd(PyObject *a, PyObject *b);
PyObject *PyNumber_InPlaceOr(PyObject *a, PyObject *b);
PyObject *PyNumber_InPlaceXor(PyObject *a, PyObject *b);

/* --- modules / import --- */
PyObject *PyModule_NewObject(PyObject *name);
const char *PyModule_GetName(PyObject *module);
PyObject *PyModule_GetFilenameObject(PyObject *module);
int PyModule_AddFunctions(PyObject *module, PyMethodDef *functions);
PyObject *PyImport_AddModule(const char *name);
PyObject *PyImport_ReloadModule(PyObject *m);

/* --- managed dict (CPython 3.11+); bridge has no managed dicts --- */
#define Py_TPFLAGS_MANAGED_DICT (1UL << 4)
void PyObject_ClearManagedDict(PyObject *obj);
int PyObject_VisitManagedDict(PyObject *obj, visitproc visit, void *arg);
int PyObject_GenericSetDict(PyObject *obj, PyObject *value, void *context);
PyObject **_PyObject_GetDictPtr(PyObject *obj);

/* --- threads (single-thread wasm: stubs at link) --- */
PyThreadState *PyThreadState_New(PyInterpreterState *interp);
void PyThreadState_Clear(PyThreadState *tstate);
void PyThreadState_DeleteCurrent(void);
PyThreadState *PyThreadState_GetUnchecked(void);
PyObject *PyThreadState_GetDict(void);
void PyEval_AcquireThread(PyThreadState *tstate);

/* --- frames / code / warnings --- */
int PyFrame_GetLineNumber(PyFrameObject *frame);
PyObject *PyEval_GetFrameLocals(void);
PyObject *PyEval_GetFrameGlobals(void);
PyObject *PyCode_GetVarnames(PyCodeObject *code);
int PyErr_WarnExplicit(PyObject *category, const char *message,
                       const char *filename, int lineno,
                       const char *module, PyObject *registry);

/* --- free-threading probe (3.14): no-op single-thread --- */
static inline void PyUnstable_EnableTryIncRef(PyObject *obj) { (void)obj; }
static inline int PyUnstable_TryIncRef(PyObject *obj) { (void)obj; return 1; }

/* --- pass 3 (full torch/csrc sweep) --- */
int PySet_Add(PyObject *set, PyObject *key);
int PySet_Contains(PyObject *set, PyObject *key);
PyObject *PyDict_Items(PyObject *d);
PyObject *PyList_GetItemRef(PyObject *list, Py_ssize_t i);
PyObject *PyModule_GetNameObject(PyObject *module);
PyFrameObject *PyEval_GetFrame(void);
int PyFrame_GetLasti(PyFrameObject *frame);
int PyCode_Addr2Line(PyCodeObject *co, int addr);
/* wasthon declares PyModule_Check as a typed function; CPython's macro
 * casts its argument (torch calls it on PyTypeObject*). */
#define PyModule_Check(op) (PyModule_Check)((PyObject *)(op))

#ifdef __cplusplus
}
#endif
#endif
