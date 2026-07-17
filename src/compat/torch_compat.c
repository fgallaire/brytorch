/* torch_compat.c — implementations for torch_compat.h, in pure C on top of
 * wasthon's existing C-API (no wasthon.js additions). Link-time companion
 * of the torch port; stubs are marked and sized for the v1 vertical slice
 * (tensor creation + ops + autograd), not for dynamo/profiler/onnx.
 *
 * v1 shortcuts (documented, revisit at run stage):
 *  - PyStructSequence_*: named tuples via collections.namedtuple; static
 *    PyTypeObject shells keep the real Python type in tp_dict. SET/GET map
 *    to PyTuple_SetItem/GetItem (bridge tuples are mutable pre-publish).
 *  - PyMemoryView_FromBuffer copies through bytes (read-only view).
 *  - Method checks probe __func__/__self__ instead of exact types.
 */
#include "torch_compat.h"
#include "structseq.h"
#include <string.h>
#include <stdio.h>

static PyObject *builtin(const char *name) {
    PyObject *mod = PyImport_ImportModule("builtins");
    if (!mod) return NULL;
    PyObject *f = PyObject_GetAttrString(mod, name);
    return f;
}

/* --- type checks / accessors --- */
int PyWeakref_Check(PyObject *o) { return PyWeakref_CheckRef(o); }

static int isinstance_of_builtin(PyObject *o, const char *name) {
    PyObject *cls = builtin(name);
    if (!cls) { PyErr_Clear(); return 0; }
    int r = PyObject_IsInstance(o, cls);
    if (r < 0) { PyErr_Clear(); return 0; }
    return r;
}
int PyFrozenSet_Check(PyObject *o) { return isinstance_of_builtin(o, "frozenset"); }
int PyGen_Check(PyObject *o)  { (void)o; return 0; }   /* generators never cross the bridge */
int PyCode_Check(PyObject *o) { (void)o; return 0; }

int PyMethod_Check(PyObject *o) {
    if (!o) return 0;
    if (!PyObject_HasAttrString(o, "__func__")) return 0;
    return PyObject_HasAttrString(o, "__self__");
}
/* PyInstanceMethod_New/Check/GET_FUNCTION are implemented bridge-side
 * (wasthon.js): pybind11 wraps every class method in an instancemethod,
 * whose DESCRIPTOR binding is what prepends self on instance access —
 * the v1 identity stub dropped self from all bound-method calls. */
PyObject *PyMethod_GET_FUNCTION(PyObject *m) {
    PyObject *f = PyObject_GetAttrString(m, "__func__");
    if (!f) PyErr_Clear();
    return f;
}

PyTypeObject PyStaticMethod_Type = {0};  /* identity compares miss -> pybind11 takes the convert path */
PyObject *PyStaticMethod_New(PyObject *func) {
    PyObject *cls = builtin("staticmethod");
    if (!cls) return NULL;
    PyObject *r = PyObject_CallFunctionObjArgs(cls, func, NULL);
    return r;
}

PyObject *PyType_GetName(PyTypeObject *type) {
    if (type->tp_name) {
        const char *dot = strrchr(type->tp_name, '.');
        return PyUnicode_FromString(dot ? dot + 1 : type->tp_name);
    }
    return PyObject_GetAttrString((PyObject *)type, "__name__");
}

/* --- objects / attributes --- */
int PyObject_HasAttr(PyObject *o, PyObject *name) {
    PyObject *v = PyObject_GetAttr(o, name);
    if (!v) { PyErr_Clear(); return 0; }
    return 1;
}
int PyObject_DelAttr(PyObject *o, PyObject *name) {
    PyObject *r = PyObject_CallMethod(o, "__delattr__", "O", name);
    if (!r) return -1;
    return 0;
}
int PyObject_DelAttrString(PyObject *o, const char *name) {
    PyObject *r = PyObject_CallMethod(o, "__delattr__", "s", name);
    if (!r) return -1;
    return 0;
}
int PyObject_GetOptionalAttrString(PyObject *o, const char *name, PyObject **out) {
    *out = PyObject_GetAttrString(o, name);
    if (!*out) { PyErr_Clear(); return 0; }
    return 1;
}
PyObject *PyDict_SetDefault(PyObject *d, PyObject *key, PyObject *dflt) {
    return PyObject_CallMethod(d, "setdefault", "OO", key, dflt);
}
Py_ssize_t PySet_Size(PyObject *s) { return PyObject_Size(s); }
int PySet_Clear(PyObject *s) {
    PyObject *r = PyObject_CallMethod(s, "clear", NULL);
    return r ? 0 : -1;
}
int PySet_Contains(PyObject *set, PyObject *key) {
    PyObject *r = PyObject_CallMethod(set, "__contains__", "O", key);
    if (!r) return -1;
    return PyObject_IsTrue(r);
}
int PySet_Add(PyObject *set, PyObject *key) {
    PyObject *r = PyObject_CallMethod(set, "add", "O", key);
    return r ? 0 : -1;
}
PyObject *PyDict_Items(PyObject *d) {
    PyObject *items = PyObject_CallMethod(d, "items", NULL);
    if (!items) return NULL;
    PyObject *lst = PySequence_List(items);
    return lst;
}
PyObject *PyList_GetItemRef(PyObject *list, Py_ssize_t i) {
    return PyList_GetItem(list, i);  /* new-vs-borrowed: no-op refcounts */
}
PyObject *PyByteArray_FromObject(PyObject *o) {
    PyObject *cls = builtin("bytearray");
    if (!cls) return NULL;
    return PyObject_CallFunctionObjArgs(cls, o, NULL);
}
PyObject *PyMemoryView_FromBuffer(const Py_buffer *view) {
    /* v1: copy through bytes (read-only). Real zero-copy needs bridge help. */
    PyObject *b = PyBytes_FromStringAndSize((const char *)view->buf, view->len);
    if (!b) return NULL;
    PyObject *mv = PyMemoryView_FromObject(b);
    return mv;
}
void PyUnicode_InternInPlace(PyObject **p) { (void)p; }
int PyCapsule_SetDestructor(PyObject *capsule, PyCapsule_Destructor destructor) {
    (void)capsule; (void)destructor;  /* capsules die with the page */
    return 0;
}

/* --- in-place number protocol: fall back to the plain ops. The bridge
 * dispatches += on Python objects itself; these C entry points are only
 * reached from torch's C++ where value semantics are acceptable v1. --- */
PyObject *PyNumber_InPlaceAdd(PyObject *a, PyObject *b)         { return PyNumber_Add(a, b); }
PyObject *PyNumber_InPlaceSubtract(PyObject *a, PyObject *b)    { return PyNumber_Subtract(a, b); }
PyObject *PyNumber_InPlaceMultiply(PyObject *a, PyObject *b)    { return PyNumber_Multiply(a, b); }
PyObject *PyNumber_InPlaceTrueDivide(PyObject *a, PyObject *b)  { return PyNumber_TrueDivide(a, b); }
PyObject *PyNumber_InPlaceLshift(PyObject *a, PyObject *b)      { return PyNumber_Lshift(a, b); }
PyObject *PyNumber_InPlaceRshift(PyObject *a, PyObject *b)      { return PyNumber_Rshift(a, b); }
PyObject *PyNumber_InPlaceAnd(PyObject *a, PyObject *b)         { return PyNumber_And(a, b); }
PyObject *PyNumber_InPlaceOr(PyObject *a, PyObject *b)          { return PyNumber_Or(a, b); }
PyObject *PyNumber_InPlaceXor(PyObject *a, PyObject *b)         { return PyNumber_Xor(a, b); }

/* --- modules / import --- */
PyObject *PyModule_NewObject(PyObject *name) {
    const char *s = PyUnicode_AsUTF8(name);
    return s ? PyModule_New(s) : NULL;
}
const char *PyModule_GetName(PyObject *module) {
    PyObject *n = PyObject_GetAttrString(module, "__name__");
    return n ? PyUnicode_AsUTF8(n) : NULL;
}
PyObject *PyModule_GetNameObject(PyObject *module) {
    return PyObject_GetAttrString(module, "__name__");
}
PyObject *PyModule_GetFilenameObject(PyObject *module) {
    PyObject *f = PyObject_GetAttrString(module, "__file__");
    if (!f) {
        /* CPython raises SystemError here; pybind11's def_submodule
         * explicitly clears ONLY that (anything else is re-thrown). */
        PyErr_Clear();
        PyErr_SetString(PyExc_SystemError, "module filename missing");
        return NULL;
    }
    return f;
}
int PyModule_AddFunctions(PyObject *module, PyMethodDef *functions) {
    for (PyMethodDef *def = functions; def && def->ml_name; def++) {
        PyObject *fn = PyCFunction_NewEx(def, NULL, NULL);
        if (!fn) return -1;
        if (PyObject_SetAttrString(module, def->ml_name, fn) < 0) return -1;
    }
    return 0;
}
/* PyImport_AddModule lives bridge-side (wasthon.js): it must register the
 * module in Brython's real import table ($B.imported) — pybind11 3.0's
 * def_submodule creates every torch._C.* submodule through it, and
 * `import torch._C._autograd` resolves against that table. */
PyObject *PyImport_ReloadModule(PyObject *m) {
    PyObject *importlib = PyImport_ImportModule("importlib");
    if (!importlib) return NULL;
    return PyObject_CallMethod(importlib, "reload", "O", m);
}

/* --- managed dict: the bridge has none --- */
void PyObject_ClearManagedDict(PyObject *obj) { (void)obj; }
int PyObject_VisitManagedDict(PyObject *obj, visitproc visit, void *arg) {
    (void)obj; (void)visit; (void)arg;
    return 0;
}
int PyObject_GenericSetDict(PyObject *obj, PyObject *value, void *context) {
    (void)context;
    return PyObject_SetAttrString(obj, "__dict__", value);
}
PyObject **_PyObject_GetDictPtr(PyObject *obj) {
    (void)obj;
    return NULL;  /* documented CPython contract: callers must handle NULL */
}

/* --- threads: single-thread wasm. ONE real static PyThreadState serves
 * every query: pybind11's gil_scoped_acquire dereferences the pointer
 * (reads/writes gilstate_counter), so a sentinel or a NULL terminates the
 * process inside error_already_set::what(). Defining
 * PyGILState_GetThisThreadState here overrides the bridge's JS sentinel
 * (native objects win over js-library definitions). --- */
static PyThreadState wasthon_torch_tstate;
static PyObject *wasthon_torch_tstate_dict;
static PyThreadState *torch_tstate(void) {
    if (!wasthon_torch_tstate.interp) {
        wasthon_torch_tstate.interp = PyInterpreterState_Get();
        wasthon_torch_tstate.gilstate_counter = 1;
    }
    return &wasthon_torch_tstate;
}
PyThreadState *PyGILState_GetThisThreadState(void) { return torch_tstate(); }
PyThreadState *PyThreadState_New(PyInterpreterState *interp) {
    (void)interp;
    return torch_tstate();
}
void PyThreadState_Clear(PyThreadState *tstate) { (void)tstate; }
void PyThreadState_DeleteCurrent(void) {}
PyThreadState *PyThreadState_GetUnchecked(void) { return torch_tstate(); }
PyObject *PyThreadState_GetDict(void) {
    if (!wasthon_torch_tstate_dict) wasthon_torch_tstate_dict = PyDict_New();
    return wasthon_torch_tstate_dict;
}
void PyEval_AcquireThread(PyThreadState *tstate) { (void)tstate; }
Py_tss_t *PyThread_tss_alloc(void) {
    static Py_tss_t keys[64];
    static int used;
    return used < 64 ? &keys[used++] : NULL;
}
static void *tss_values[64];
int PyThread_tss_create(Py_tss_t *key) { (void)key; return 0; }
void PyThread_tss_delete(Py_tss_t *key) { (void)key; }
int PyThread_tss_set(Py_tss_t *key, void *value) {
    if (*key < 0 || *key >= 64) return -1;
    tss_values[*key] = value;
    return 0;
}
void *PyThread_tss_get(Py_tss_t *key) {
    if (*key < 0 || *key >= 64) return NULL;
    return tss_values[*key];
}

/* --- frames / code / warnings: no C-level frame introspection ---
 * GetCode/GetBack/GetFrame are declared by wasthon.h but had no
 * implementation anywhere (never linked before torch). */
PyCodeObject *PyFrame_GetCode(PyFrameObject *frame) { (void)frame; return NULL; }
PyFrameObject *PyFrame_GetBack(PyFrameObject *frame) { (void)frame; return NULL; }
PyFrameObject *PyThreadState_GetFrame(PyThreadState *tstate) { (void)tstate; return NULL; }
int PyFrame_GetLineNumber(PyFrameObject *frame) { (void)frame; return 0; }
int PyFrame_GetLasti(PyFrameObject *frame) { (void)frame; return -1; }
int PyCode_Addr2Line(PyCodeObject *co, int addr) { (void)co; (void)addr; return -1; }
PyFrameObject *PyEval_GetFrame(void) { return NULL; }
PyObject *PyEval_GetFrameLocals(void) { return PyDict_New(); }
PyObject *PyEval_GetFrameGlobals(void) { return PyDict_New(); }
PyObject *PyCode_GetVarnames(PyCodeObject *code) { (void)code; return PyTuple_New(0); }
int PyErr_WarnExplicit(PyObject *category, const char *message,
                       const char *filename, int lineno,
                       const char *module, PyObject *registry) {
    (void)filename; (void)lineno; (void)module; (void)registry;
    return PyErr_WarnEx(category, message, 1);
}
int _PyEval_SliceIndex(PyObject *v, Py_ssize_t *pi) {
    if (!v || v == Py_None) return 1;
    if (!PyIndex_Check(v)) {
        PyErr_SetString(PyExc_TypeError,
                        "slice indices must be integers or None or have an "
                        "__index__ method");
        return 0;
    }
    Py_ssize_t x = PyNumber_AsSsize_t(v, NULL);
    if (x == -1 && PyErr_Occurred()) return 0;
    *pi = x;
    return 1;
}

/* --- struct sequences: v1 = collections.namedtuple. The static
 * PyTypeObject shells used by generated return_types keep the real Python
 * type in tp_dict; New instantiates it filled with None and SET_ITEM
 * pokes values through the bridge's pre-publish tuple mutability. --- */
const char * const PyStructSequence_UnnamedField = "unnamed field";

static PyObject *structseq_make_type(PyStructSequence_Desc *desc) {
    PyObject *collections = PyImport_ImportModule("collections");
    if (!collections) return NULL;
    const char *dot = strrchr(desc->name, '.');
    PyObject *fields = PyList_New(0);
    if (!fields) return NULL;
    for (int i = 0; i < desc->n_in_sequence; i++) {
        const char *fn = desc->fields[i].name;
        char buf[32];
        if (!fn || fn == PyStructSequence_UnnamedField) {
            snprintf(buf, sizeof buf, "_%d", i);
            fn = buf;
        }
        PyObject *f = PyUnicode_FromString(fn);
        if (!f || PyList_Append(fields, f) < 0) return NULL;
    }
    PyObject *t = PyObject_CallMethod(collections, "namedtuple", "sO",
                                      dot ? dot + 1 : desc->name, fields);
    return t;
}

PyTypeObject *PyStructSequence_NewType(PyStructSequence_Desc *desc) {
    return (PyTypeObject *)structseq_make_type(desc);
}
extern int wasthon_alias_ptr(void *ptr, PyObject *obj);
int PyStructSequence_InitType2(PyTypeObject *type, PyStructSequence_Desc *desc) {
    PyObject *t = structseq_make_type(desc);
    if (!t) return -1;
    type->tp_name = desc->name;
    type->tp_dict = t;  /* shell keeps the real type; New() looks it up */
    /* generated code stores the SHELL pointer in module dicts
     * (PyModule_AddObject) — make it unwrap to the real class */
    wasthon_alias_ptr(type, t);
    return 0;
}
void PyStructSequence_InitType(PyTypeObject *type, PyStructSequence_Desc *desc) {
    (void)PyStructSequence_InitType2(type, desc);
}
PyObject *PyStructSequence_New(PyTypeObject *type) {
    PyObject *real = type->tp_dict ? type->tp_dict : (PyObject *)type;
    PyObject *fields = PyObject_GetAttrString(real, "_fields");
    if (!fields) return NULL;
    Py_ssize_t n = PyObject_Size(fields);
    PyObject *nones = PyTuple_New(n);
    if (!nones) return NULL;
    for (Py_ssize_t i = 0; i < n; i++) PyTuple_SetItem(nones, i, Py_None);
    /* namedtuple(*values) */
    PyObject *star = PyObject_CallMethod(real, "_make", "O", nones);
    return star;
}
PyObject *PyStructSequence_GetItem(PyObject *self, Py_ssize_t i) {
    return PyTuple_GetItem(self, i);
}
void PyStructSequence_SetItem(PyObject *self, Py_ssize_t i, PyObject *value) {
    PyTuple_SetItem(self, i, value);
}

/* --- pybind11 needs &PyProperty_Type callable (enum_base::init builds
 * enum properties through it). Binding it inside the SHARED wasthon_init
 * is forbidden (see wasthon.c: latent sqlite3 OOB regression); the torch
 * page binds it after wasthon_init through this accessor. --- */
#include <emscripten.h>
EMSCRIPTEN_KEEPALIVE
PyTypeObject *wasthon_torch_property_addr(void) { return &PyProperty_Type; }
EMSCRIPTEN_KEEPALIVE
PyTypeObject *wasthon_torch_staticmethod_addr(void) { return &PyStaticMethod_Type; }
