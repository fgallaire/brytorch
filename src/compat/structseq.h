/* structseq.h — PyStructSequence declarations for the torch port.
 * torch/csrc/python_headers.h includes <structseq.h> unconditionally;
 * torch uses struct sequences for torch.return_types (min/max/sort...).
 * Declarations only: the implementations live bridge-side (wasthon).
 * Kept out of wasthon.h until the torch port lands (same pattern as
 * scipy.special's sp_compat.h in numbry). */
#ifndef WASTHON_STRUCTSEQ_H
#define WASTHON_STRUCTSEQ_H
#ifdef __cplusplus
extern "C" {
#endif

typedef PyTupleObject PyStructSequence;

typedef struct PyStructSequence_Field {
    const char *name;
    const char *doc;
} PyStructSequence_Field;

typedef struct PyStructSequence_Desc {
    const char *name;
    const char *doc;
    PyStructSequence_Field *fields;
    int n_in_sequence;
} PyStructSequence_Desc;

extern const char * const PyStructSequence_UnnamedField;

PyTypeObject *PyStructSequence_NewType(PyStructSequence_Desc *desc);
int  PyStructSequence_InitType2(PyTypeObject *type, PyStructSequence_Desc *desc);
void PyStructSequence_InitType(PyTypeObject *type, PyStructSequence_Desc *desc);
PyObject *PyStructSequence_New(PyTypeObject *type);
PyObject *PyStructSequence_GetItem(PyObject *self, Py_ssize_t i);
void PyStructSequence_SetItem(PyObject *self, Py_ssize_t i, PyObject *value);
#define PyStructSequence_GET_ITEM PyStructSequence_GetItem
#define PyStructSequence_SET_ITEM PyStructSequence_SetItem

#ifdef __cplusplus
}
#endif
#endif
