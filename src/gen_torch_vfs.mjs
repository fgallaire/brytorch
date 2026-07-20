#!/usr/bin/env node
// Build ~/wastensor/build/torch_vfs.js — torch's pure-Python layer as a
// Brython VFS (pattern: numbry gen_scipy_vfs.mjs). Pairs with
// pytorch/build-wasm/npth.mjs (the linked torch._C).
//
// v1 scope: everything under torch/ EXCEPT the heavy non-portable subtrees
// (_inductor, distributed, _dynamo, onnx, testing/_internal) which become
// PEP-562 raise-at-use stubs. torch/version.py (setup.py-generated, absent
// from a git clone) is synthesized; typing_extensions rides along from the
// system python.
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
const require = createRequire(import.meta.url);
const fs = require('fs'), path = require('path');
const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const PT = path.join(ROOT, 'pytorch');
const OUT = path.join(ROOT, 'build', 'torch_vfs.js');

let scripts = { $timestamp: Date.now() };
let n = 0, bytes = 0;

const EXCLUDE_DIRS = new Set([
  'torch._inductor', 'torch.distributed', 'torch._dynamo', 'torch.onnx',
  // testing._internal ships REAL (the official-suite dashboard imports
  // common_utils/common_device_type); only its distributed subtree stays out
  'torch.testing._internal.distributed',
  // export/compile infrastructure — out of the v1 slice like dynamo (its
  // import chain also trips a pybind11 holder cast, see PYTORCH_PORT.md)
  'torch.export', 'torch._export', 'torch._higher_order_ops',
  'torch.utils._debug_mode',
]);

const PATCH = {
  // _make_grads pulls expect_true/sym_eq from symbolic_shapes, whose module
  // level does `import sympy` (not shipped). Concrete-shape equivalents are
  // exact for the eager build: expect_true is a guard passthrough, sym_eq a
  // structural size compare.
  'torch.autograd': (s) => s.replace(
    '            from torch.fx.experimental.symbolic_shapes import expect_true, sym_eq',
    '            try:  # wasthon: no sympy — concrete-shape equivalents\n'
    + '                from torch.fx.experimental.symbolic_shapes import expect_true, sym_eq\n'
    + '            except ModuleNotFoundError:\n'
    + '                def expect_true(x):\n'
    + '                    return x\n'
    + '                def sym_eq(a, b):\n'
    + '                    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):\n'
    + '                        return len(a) == len(b) and all(sym_eq(x, y) for x, y in zip(a, b))\n'
    + '                    return a == b'),
  // No shared-object preloading in the browser: the wasm module is already
  // linked. ctypes never gets exercised on this path.
  'torch._utils_internal': (s) => s
    .replace(/^USE_GLOBAL_DEPS\s*=\s*True/m, 'USE_GLOBAL_DEPS = False'),
  // v1: torch swaps sys.modules['torch.backends'] for a GenericModule
  // wrapper; Brython's import engine then finds no __path__ on the parent
  // and submodule imports die. Keep the real module (wrapper only serves
  // VERBOSE flag propagation).
  'torch.backends': (s) => s.replace(
    'sys.modules[__name__] = GenericModule(sys.modules[__name__], __name__)',
    'pass  # wasthon: keep the real module object'),
  // v1: this dataclass's PEP-649 annotations crash Brython's annotate
  // evaluation (undefined leak, name not yet isolated); plain class with
  // the same two-positional constructor is behaviorally identical
  'torch._ops': (s) => s.replace(
    '@dataclass\nclass _PyObjectDispatcher(Generic[_P, _T]):',
    'class _PyObjectDispatcher(Generic[_P, _T]):\n' +
    '    def __init__(self, dispatch, redispatch):\n' +
    '        self.dispatch = dispatch\n' +
    '        self.redispatch = redispatch'),
  // v1: Brython method-wrappers (descriptor.__get__) aren't hashable yet
  // (BRYTHON_FIX pending); the passthrough set works with the descriptors
  // themselves for everything the vertical slice exercises.
  'torch._prims_common': (s) => s.replace(/\.__get__,/g, ','),
  // DEBT: pybind11 refuses the Argument.type TypePtr cast ("non-held to
  // held instance", c10::SingletonOrSharedTypePtr) in this build — the
  // kwarg-only-tensor validation is introspective-only, and no v1 custom
  // op has kwarg-only tensors; revisit the caster if a.type is needed live
  'torch._library.utils': (s) => s.replace(
    'def has_kwarg_only_tensors(schema: _C.FunctionSchema):\n    for a in schema.arguments:',
    'def has_kwarg_only_tensors(schema: _C.FunctionSchema):\n' +
    '    try:  # wasthon: a.type cast unsupported (see generator note)\n' +
    '        return _has_kwarg_only_tensors(schema)\n' +
    '    except RuntimeError:\n' +
    '        return False\n' +
    'def _has_kwarg_only_tensors(schema: _C.FunctionSchema):\n    for a in schema.arguments:')
  .replace(
    'def is_tensor_like_type(typ: Any) -> bool:\n    return typ == _C.TensorType.get() or typ == _C.OptionalType(_C.TensorType.get())',
    'def is_tensor_like_type(typ: Any) -> bool:\n' +
    '    try:  # wasthon: holder cast can fail on cached unheld Type instances\n' +
    '        return typ == _C.TensorType.get() or typ == _C.OptionalType(_C.TensorType.get())\n' +
    '    except RuntimeError:\n' +
    '        return False')
  .replace(
    'def is_tensorlist_like_type(typ: Any) -> bool:\n    return (\n        typ == _C.ListType(_C.TensorType.get())',
    'def is_tensorlist_like_type(typ: Any) -> bool:\n' +
    '    try:  # wasthon: holder cast can fail on cached unheld Type instances\n' +
    '        return _is_tensorlist_like_type(typ)\n' +
    '    except RuntimeError:\n' +
    '        return False\n' +
    'def _is_tensorlist_like_type(typ: Any) -> bool:\n    return (\n        typ == _C.ListType(_C.TensorType.get())'),
  // no fork in the browser — CPython-Emscripten has no os.register_at_fork
  // either; registering the callback is meaningless, not an error
  'torch.multiprocessing._atfork': (s) => s.replace(
    '    def _register(func):\n        os.register_at_fork(after_in_child=func)',
    '    def _register(func):\n' +
    '        if hasattr(os, "register_at_fork"):  # wasthon: no fork in wasm\n' +
    '            os.register_at_fork(after_in_child=func)'),
  // Brython module __dict__ leaks internal $-prefixed keys ($annotations
  // bookkeeping, raw JS objects) — Python identifiers can never contain
  // '$', so skipping them in the config walk is semantically neutral
  // (BRYTHON_FIX candidate: filter $-keys from namespace dict iteration)
  'torch.utils._config_module': (s) => s.replace(
    'for key, value in list(source.__dict__.items()):\n            if (\n                key.startswith("__")',
    'for key, value in list(source.__dict__.items()):\n            if (\n                "$" in key\n                or key.startswith("__")'),
  // Brython's empty-spec f-string shortcuts to str() and never consults
  // Tensor.__format__ (whose 0-dim path does .item()) — repr of any tensor
  // recursed to death; call __format__ explicitly (BRYTHON_FIX candidate)
  'torch._tensor_str': (s) => s
    .replace('                value_str = f"{value}"',
             '                value_str = type(value).__format__(value, "")')
    .replace('            ret = f"{value}"',
             '            ret = type(value).__format__(value, "")'),
  // official-suite dashboard: numpy is OPTIONAL (TEST_NUMPY gates the numpy
  // paths — pytorch-sanctioned numpy-less runs); cpp_extension pulls
  // setuptools (absent in the browser) and only serves compiled-extension
  // tests; the onnx symbolic registrars come from the stub below.
  'torch.testing._internal.common_utils': (s) => s
    .replace('import expecttest\nimport numpy as np\n',
             'import expecttest\n'
             + 'try:  # wasthon: numpy optional (TEST_NUMPY gates its tests)\n'
             + '    import numpy as np\n'
             + 'except ImportError:\n'
             + '    np = None\n')
    .replace('numpy_to_torch_dtype_dict = {',
             'numpy_to_torch_dtype_dict = {} if np is None else {')
    .replace('from torch.utils import cpp_extension',
             'try:  # wasthon: cpp_extension needs setuptools (no browser build)\n'
             + '    from torch.utils import cpp_extension\n'
             + 'except ImportError:\n'
             + '    cpp_extension = None')
    .replace(/platform\.mac_ver\(\)/g,
             // Brython's platform module (which its own os depends on —
             // vendoring CPython's creates an os.uname<->platform.uname
             // cycle) has no mac_ver; CPython's non-Mac return shape
             'getattr(platform, "mac_ver", lambda: ("", ("", "", ""), ""))()')
    .replace('    if TEST_NUMPY:\n        np.random.seed(seed)',
             // NumBry's numpy.random (Cython) cannot initialize while the
             // torch bridge owns the shared $B hooks (dual-runtime debt);
             // numpy CORE works — degrade the seed call, not the test run
             '    if TEST_NUMPY:\n'
             + '        try:  # wasthon: numpy.random unavailable (dual-runtime debt)\n'
             + '            np.random.seed(seed)\n'
             + '        except (AttributeError, ImportError, ModuleNotFoundError):\n'
             + '            pass'),
  'torch.testing._internal.common_device_type': (s) => s
    .replace(/platform\.mac_ver\(\)/g,
             'getattr(platform, "mac_ver", lambda: ("", ("", "", ""), ""))()'),
  // no shm manager binary in the browser (libshm is stubbed wasm-side)
  'torch': (s) => s
    // torch.__getattr__ lazily imports submodules RELATIVELY; Brython's
    // importlib._resolve_name rejects it ("beyond top-level package",
    // BRYTHON_FIX candidate) — the absolute form is semantically identical
    .replace('return importlib.import_module(f".{name}", __name__)',
             'return importlib.import_module(f"{__name__}.{name}")')
    // _native (out-of-tree override ops) trips on str(FunctionSchema)
    // lacking the "ns::" prefix in this build; nothing in the v1 slice
    // uses these ops — degrade the LAST import of __init__ gracefully
    .replace('import torch._native',
             'try:  # wasthon: _native override ops out of the v1 slice\n'
             + '    import torch._native\n'
             + 'except Exception as _native_err:\n'
             + '    import warnings as _w\n'
             + '    _w.warn(f"torch._native disabled (wasm v1): {_native_err}")')
    .replace('def _manager_path() -> bytes:',
             'def _manager_path() -> bytes:\n    return b""  # wasthon: no shm manager\n\ndef _manager_path_unused() -> bytes:')
    // numpy lives in a SIBLING wasm (NumBry) — the C-level tensor<->ndarray
    // bridge (USE_NUMPY) cannot share memory across the two heaps. Serve
    // VALUE-COPY equivalents in Python: t.numpy() and torch.from_numpy()
    // preserve dtype and values; view/aliasing semantics are honestly lost
    // (tests asserting shared storage fail as they should).
    + '\n\ntry:  # wasthon: value-copy numpy interop (numpy is a sibling wasm)\n'
    + '    import numpy as _wasthon_np\n'
    + 'except ImportError:\n'
    + '    _wasthon_np = None\n'
    + 'if _wasthon_np is not None:\n'
    + '    def _wasthon_tensor_numpy(self, *, force=False):\n'
    + '        t = self.detach() if self.requires_grad else self\n'
    + '        if t.is_conj() or t.is_neg():\n'
    + '            if not force:\n'
    // upstream's two distinct messages (tensor_numpy.cpp) — the suite
    // asserts on "has conjugate bit set" / "has negative bit set"
    + '                if t.is_conj():\n'
    + '                    raise RuntimeError("Can\\\'t call numpy() on Tensor that has conjugate bit set. Use tensor.resolve_conj().numpy() instead.")\n'
    + '                raise RuntimeError("Can\\\'t call numpy() on Tensor that has negative bit set. Use tensor.resolve_neg().numpy() instead.")\n'
    + '            t = t.resolve_conj().resolve_neg()\n'
    + '        _np_dt = str(t.dtype).replace("torch.", "")\n'
    + '        if t.numel() == 0:\n'
    + '            return _wasthon_np.empty(tuple(t.shape), dtype=_np_dt)\n'
    + '        return _wasthon_np.array(t.tolist(), dtype=_np_dt)\n'
    + '    Tensor.numpy = _wasthon_tensor_numpy\n'
    + '    def _wasthon_torch_dtype(np_dtype):\n'
    + '        return getattr(sys.modules["torch"], str(_wasthon_np.dtype(np_dtype)))\n'
    + '    def from_numpy(arr):\n'
    + '        a = _wasthon_np.asarray(arr)\n'
    + '        if a.size == 0:\n'
    + '            return empty(tuple(a.shape), dtype=_wasthon_torch_dtype(a.dtype))\n'
    + '        return tensor(a.tolist(), dtype=_wasthon_torch_dtype(a.dtype))\n'
    // Tensor-op-ndarray goes through torch's __torch_function__ dispatch
    // BEFORE the C++ arg parser (which would read the foreign nprnd handle
    // as npth memory and trap "index out of bounds"). Convert ndarray
    // operands value-copy for Tensor METHODS (operators dispatch as e.g.
    // method 'add' of TensorBase); plain torch.* functions keep failing
    // (NotImplemented -> TypeError), mirroring upstream's open numpy-arg
    // bug (pytorch#36363) that test_type_promotion encodes.
    + '    def _wasthon_ndarray_tf(cls, func, types, args=(), kwargs=None):\n'
    + '        if kwargs is None:\n'
    + '            kwargs = {}\n'
    + '        if getattr(func, "__objclass__", None) is None:\n'
    + '            return NotImplemented\n'
    + '        def _cv(x):\n'
    + '            return from_numpy(x) if isinstance(x, _wasthon_np.ndarray) else x\n'
    + '        return func(*[_cv(x) for x in args], **{k: _cv(v) for k, v in kwargs.items()})\n'
    + '    try:\n'
    + '        _wasthon_np.ndarray.__torch_function__ = classmethod(_wasthon_ndarray_tf)\n'
    + '    except Exception:\n'
    + '        pass\n'
    // torch.as_tensor/tensor(ndarray) take the C++ DLPack route — reading
    // the foreign capsule ACROSS the two wasm heaps yields garbage
    // ("unsupported DLPack capsule major version: <random>"). Divert the
    // ndarray case to the value-copy converter at Python level; the
    // comparison harness (torch.testing._comparison) as_tensor's every
    // numpy input, so this gates most against-numpy tests.
    // numpy SCALARS (np.generic — np.trace/np.sum results) are foreign
    // handles too: torch.tensor(np.int64(5)) trapped the same way. Convert
    // via .item() with the dtype preserved (0-d tensor, like CPython).
    + '    def _wasthon_scalar_tensor(s, dtype=None):\n'
    + '        t = _wasthon_c_tensor(s.item(),\n'
    + '                              dtype=_wasthon_torch_dtype(s.dtype))\n'
    + '        return t.to(dtype) if dtype is not None else t\n'
    // dtype=float/int/bool/complex (Python types): upstream maps them in
    // toScalarType via POINTER comparison with &PyFloat_Type & co — the
    // bridge's canonical class handles are different addresses, so the C
    // rejected them ("dtype must be torch.dtype, not type"). Map at the
    // Python boundary with upstream's exact table.
    + '    import builtins as _wasthon_bt\n'
    + '    _wasthon_pydt = {_wasthon_bt.float: float64, _wasthon_bt.int: int64,\n'
    + '                     _wasthon_bt.bool: getattr(sys.modules["torch"], "bool"),\n'
    + '                     _wasthon_bt.complex: complex128}\n'
    + '    def _wasthon_fix_dtype(d):\n'
    + '        try:\n'
    + '            return _wasthon_pydt.get(d, d)\n'
    + '        except TypeError:\n'
    + '            return d\n'
    + '    _wasthon_c_arange = arange\n'
    + '    def arange(*a, **kw):\n'
    + '        if kw.get("dtype") is not None:\n'
    + '            kw["dtype"] = _wasthon_fix_dtype(kw["dtype"])\n'
    + '        return _wasthon_c_arange(*a, **kw)\n'
    + '    _wasthon_c_randn = randn\n'
    + '    def randn(*a, **kw):\n'
    + '        if kw.get("dtype") is not None:\n'
    + '            kw["dtype"] = _wasthon_fix_dtype(kw["dtype"])\n'
    + '        return _wasthon_c_randn(*a, **kw)\n'
    + '    _wasthon_c_as_tensor = as_tensor\n'
    + '    def as_tensor(data, dtype=None, device=None):\n'
    + '        dtype = _wasthon_fix_dtype(dtype)\n'
    + '        if isinstance(data, _wasthon_np.ndarray):\n'
    + '            t = from_numpy(data)\n'
    + '            return t.to(dtype) if dtype is not None else t\n'
    + '        if isinstance(data, _wasthon_np.generic):\n'
    + '            return _wasthon_scalar_tensor(data, dtype)\n'
    + '        return _wasthon_c_as_tensor(data, dtype=dtype, device=device)\n'
    + '    _wasthon_c_tensor = tensor\n'
    + '    def tensor(data, *args, **kw):\n'
    + '        if kw.get("dtype") is not None:\n'
    + '            kw["dtype"] = _wasthon_fix_dtype(kw["dtype"])\n'
    + '        if isinstance(data, _wasthon_np.ndarray):\n'
    + '            t = from_numpy(data)\n'
    + '            if kw.get("dtype") is not None:\n'
    + '                t = t.to(kw["dtype"])\n'
    + '            if kw.get("requires_grad"):\n'
    + '                t.requires_grad_(True)\n'
    + '            return t\n'
    + '        if isinstance(data, _wasthon_np.generic):\n'
    + '            t = _wasthon_scalar_tensor(data, kw.get("dtype"))\n'
    + '            if kw.get("requires_grad"):\n'
    + '                t.requires_grad_(True)\n'
    + '            return t\n'
    + '        return _wasthon_c_tensor(data, *args, **kw)\n',
};

function add(mod, src, isInit) {
  if (PATCH[mod]) src = PATCH[mod](src);
  scripts[mod] = ['.py', src, [], !!isInit];
  n++; bytes += src.length;
}

function walk(dir, prefix) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (e.isDirectory()) {
      const sub = prefix + '.' + e.name;
      if (e.name === '__pycache__' || e.name === 'test' || e.name === 'tests'
          || EXCLUDE_DIRS.has(sub)) continue;
      walk(path.join(dir, e.name), sub);
      continue;
    }
    if (!e.name.endsWith('.py')) continue;
    const isInit = e.name === '__init__.py';
    const mod = isInit ? prefix : prefix + '.' + e.name.replace(/\.py$/, '');
    add(mod, fs.readFileSync(path.join(dir, e.name), 'utf8'), isInit);
  }
}

walk(path.join(PT, 'torch'), 'torch');
walk(path.join(PT, 'torchgen'), 'torchgen');  // torch.utils._python_dispatch imports it

// setup.py-generated, absent from a git clone
add('torch.version',
  '__version__ = "2.14.0a0+wasthon"\n' +
  'debug = False\ncuda = None\nhip = None\nxpu = None\ngit_version = ""\n',
  false);

// PEP-562 raise-at-use stubs for the excluded subtrees. Brython consults
// module __getattr__ on attribute access but NOT on from-import fallback,
// so hard `from torch.X import y` sites inside the kept scope will need
// explicit names here as they surface.
for (const mod of ['torch._inductor', 'torch.distributed', 'torch._dynamo',
                   'torch.onnx', 'torch.export', 'torch._export',
                   'torch.utils._debug_mode']) {
  add(mod,
    '# wasthon v1 stub: subsystem not in this wasm build\n' +
    'def __getattr__(name):\n' +
    '    raise ImportError("' + mod + ' is not in this wasm build (v1)")\n',
    true);
}
// distributed probes used by torch.__init__/serialization.
// `rpc` is exposed inline: nn/parallel/distributed.py does
// `if dist.rpc.is_available():` at module level, and the __getattr__
// raise would kill the import before the guard gets to say no.
// name-reporting: the suites reach _inductor from several sites — say WHICH
scripts['torch._inductor'][1] =
  '# wasthon v1 stub: subsystem not in this wasm build\n' +
  'def __getattr__(name):\n' +
  '    if name == "config":\n' +
  '        import importlib\n' +
  '        mod = importlib.import_module("torch._inductor.config")\n' +
  '        globals()["config"] = mod\n' +
  '        return mod\n' +
  '    raise ImportError("torch._inductor is not in this wasm build (v1): attr " + name)\n';

// config flags read (and @patch-ed) by the harness — same inert shape as
// the dynamo stub's config
add('torch._inductor.config', [
  'class _PatchCtx:',
  '    def __call__(self, fn=None):',
  '        return fn if fn is not None else self',
  '    def __enter__(self):',
  '        return self',
  '    def __exit__(self, *exc):',
  '        return False',
  'def patch(*a, **k):',
  '    return _PatchCtx()',
  'def __getattr__(name):',
  '    return False',
  ''].join('\n'), false);

// FUNCTIONAL no-op stub: common_utils' TestCase machinery drives dynamo
// controls around every test (reset / set_stance / config.suppress_errors) —
// in an eager-only build these are inert, not errors. Unknown attrs still
// raise WITH the attr name (diagnosis).
scripts['torch._dynamo'][1] = [
  '# wasthon v1: no compiler in this build - dynamo controls are inert no-ops',
  'class _PatchCtx:',
  '    def __call__(self, fn=None):',
  '        return fn if fn is not None else self',
  '    def __enter__(self):',
  '        return self',
  '    def __exit__(self, *exc):',
  '        return False',
  'class _Config:',
  '    def __getattr__(self, name):',
  '        return False',
  '    def patch(self, *a, **k):',
  '        return _PatchCtx()',
  'config = _Config()',
  'def reset(*a, **k):',
  '    pass',
  'def reset_code_caches(*a, **k):',
  '    pass',
  'class _Stance:',
  '    def __call__(self, fn=None):',
  '        return fn',
  '    def __enter__(self):',
  '        return self',
  '    def __exit__(self, *exc):',
  '        return False',
  'def set_stance(*a, **k):',
  '    return _Stance()',
  'def is_compiling():',
  '    return False',
  'def disable(fn=None, recursive=True, *a, **k):',
  '    if fn is None:',
  '        return lambda f: f',
  '    return fn',
  'def graph_break():',
  '    pass',
  'def mark_dynamic(*a, **k):',
  '    pass',
  'def maybe_mark_dynamic(*a, **k):',
  '    pass',
  'def mark_static(*a, **k):',
  '    pass',
  'class _Exc:',
  '    class BackendCompilerFailed(RuntimeError):',
  '        pass',
  'exc = _Exc()',
  'def __getattr__(name):',
  '    raise ImportError("torch._dynamo is not in this wasm build (v1): attr " + name)',
  ''].join('\n');

// common_utils does `from torch.onnx import register_custom_op_symbolic,
// unregister_custom_op_symbolic` — symbolic registration is inert without
// the onnx exporter, so serve them as no-ops on the stub
scripts['torch.onnx'][1] =
  '# wasthon v1 stub: subsystem not in this wasm build\n' +
  'def register_custom_op_symbolic(symbolic_name, symbolic_fn, opset_version):\n' +
  '    pass\n' +
  'def unregister_custom_op_symbolic(symbolic_name, opset_version):\n' +
  '    pass\n' +
  'def __getattr__(name):\n' +
  '    raise ImportError("torch.onnx is not in this wasm build (v1)")\n';

scripts['torch.distributed'][1] =
  'def is_available():\n    return False\n' +
  'class _RpcProbe:\n' +
  '    @staticmethod\n' +
  '    def is_available():\n' +
  '        return False\n' +
  'rpc = _RpcProbe()\n' +
  'def __getattr__(name):\n' +
  '    raise ImportError("torch.distributed is not in this wasm build (v1)")\n';

// ctypes browser stub (Brython has none; torch only exercises it on the
// shared-object preload path, which USE_GLOBAL_DEPS=False disarms)
add('ctypes', [
  'RTLD_GLOBAL = 0',
  'RTLD_LOCAL = 0',
  'DEFAULT_MODE = 0',
  'class ArgumentError(Exception): pass',
  'class CDLL:',
  '    def __init__(self, name=None, mode=0, *a, **k):',
  '        raise OSError("ctypes.CDLL is not available in the browser: " + repr(name))',
  'PyDLL = WinDLL = OleDLL = CDLL',
  'class LibraryLoader:',
  '    def __init__(self, dlltype):',
  '        self._dlltype = dlltype',
  '    def LoadLibrary(self, name):',
  '        raise OSError("ctypes dll loading is not available in the browser")',
  '    def __getattr__(self, name):',
  '        raise OSError("ctypes dll loading is not available in the browser")',
  'cdll = LibraryLoader(CDLL)',
  'pydll = LibraryLoader(CDLL)',
  'def WinError(*a, **k):',
  '    raise OSError("ctypes.WinError is not available")',
  'class _CFuncPtr: pass',
  'def CFUNCTYPE(*a, **k):',
  '    return type("CFunctionType", (_CFuncPtr,), {})',
  'PYFUNCTYPE = CFUNCTYPE',
  'class _SimpleCData:',
  '    def __init__(self, value=None):',
  '        self.value = value',
  'class c_void_p(_SimpleCData): pass',
  'class c_char_p(_SimpleCData): pass',
  'class c_wchar_p(_SimpleCData): pass',
  'class c_bool(_SimpleCData): pass',
  'class c_byte(_SimpleCData): pass',
  'class c_ubyte(_SimpleCData): pass',
  'class c_short(_SimpleCData): pass',
  'class c_ushort(_SimpleCData): pass',
  'class c_int(_SimpleCData): pass',
  'class c_uint(_SimpleCData): pass',
  'class c_long(_SimpleCData): pass',
  'class c_ulong(_SimpleCData): pass',
  'class c_longlong(_SimpleCData): pass',
  'class c_ulonglong(_SimpleCData): pass',
  'class c_size_t(_SimpleCData): pass',
  'class c_ssize_t(_SimpleCData): pass',
  'class c_int8(_SimpleCData): pass',
  'class c_uint8(_SimpleCData): pass',
  'class c_int16(_SimpleCData): pass',
  'class c_uint16(_SimpleCData): pass',
  'class c_int32(_SimpleCData): pass',
  'class c_uint32(_SimpleCData): pass',
  'class c_int64(_SimpleCData): pass',
  'class c_uint64(_SimpleCData): pass',
  'class c_float(_SimpleCData): pass',
  'class c_double(_SimpleCData): pass',
  'class c_char(_SimpleCData): pass',
  'class c_wchar(_SimpleCData): pass',
  'class py_object(_SimpleCData): pass',
  'class Structure: pass',
  'class Union: pass',
  'class Array: pass',
  'class _Ptr: pass',
  'def POINTER(*a, **k):',
  '    return _Ptr',
  'def pointer(*a, **k):',
  '    return _Ptr()',
  'def byref(obj, offset=0):',
  '    return obj',
  'def sizeof(obj):',
  '    return 0',
  'def addressof(obj):',
  '    return 0',
  'def alignment(obj):',
  '    return 0',
  'def memmove(dst, src, count):',
  '    raise OSError("ctypes.memmove is not available in the browser")',
  'def memset(dst, c, count):',
  '    raise OSError("ctypes.memset is not available in the browser")',
  'def cast(obj, typ):',
  '    return typ() if callable(typ) else obj',
  'def create_string_buffer(init, size=None):',
  '    return bytearray(init if isinstance(init, int) else len(init))',
  'def string_at(ptr, size=-1):',
  '    raise OSError("ctypes.string_at is not available in the browser")',
  'def get_last_error():',
  '    return 0',
  ''].join('\n'), true);
add('ctypes.util', [
  'def find_library(name):',
  '    return None',
  ''].join('\n'), false);
add('ctypes.wintypes', '', false);
add('ctypes.macholib', '', true);

// mmap: no browser equivalent; torch.serialization imports it at module
// level but only exercises it for torch.load(mmap=True)
add('mmap', [
  'ACCESS_READ = 1',
  'ACCESS_WRITE = 2',
  'ACCESS_COPY = 3',
  'ACCESS_DEFAULT = 0',
  'ALLOCATIONGRANULARITY = 4096',
  'MAP_SHARED = 1',
  'MAP_PRIVATE = 2',
  'MAP_ANONYMOUS = 32',
  'MAP_ANON = 32',
  'MAP_DENYWRITE = 2048',
  'MAP_EXECUTABLE = 4096',
  'MAP_POPULATE = 32768',
  'PROT_READ = 1',
  'PROT_WRITE = 2',
  'PROT_EXEC = 4',
  'PAGESIZE = 4096',
  'class error(OSError): pass',
  'class mmap:',
  '    def __init__(self, *a, **k):',
  '        raise OSError("mmap is not available in the browser")',
  ''].join('\n'), false);

// multiprocessing: Brython's copy dies at import (process.py os glue);
// single-process browser semantics served as a stub
add('multiprocessing', [
  'def cpu_count():',
  '    return 1',
  'class _Process:',
  '    name = "MainProcess"',
  '    pid = 1',
  '    daemon = False',
  '    authkey = b""',
  '    def is_alive(self):',
  '        return True',
  'def current_process():',
  '    return _Process()',
  'def active_children():',
  '    return []',
  'def get_start_method(allow_none=False):',
  '    return "spawn"',
  'def set_start_method(method, force=False):',
  '    pass',
  'def get_all_start_methods():',
  '    return ["spawn"]',
  'def get_context(method=None):',
  '    import sys as _s',
  '    return _s.modules[__name__]',
  'def _browser_unavailable(*a, **k):',
  '    raise OSError("multiprocessing is not available in the browser")',
  'Process = Pool = Queue = SimpleQueue = JoinableQueue = _browser_unavailable',
  'Lock = RLock = Semaphore = BoundedSemaphore = Condition = Event = Barrier = _browser_unavailable',
  'Manager = Pipe = Value = Array = _browser_unavailable',
  'class ProcessError(Exception): pass',
  'class BufferTooShort(ProcessError): pass',
  'class AuthenticationError(ProcessError): pass',
  'class TimeoutError(ProcessError): pass',
  // torch/multiprocessing does `from multiprocessing import *` then
  // `__all__ += multiprocessing.__all__`
  '__all__ = ["Array", "AuthenticationError", "Barrier", "BoundedSemaphore",',
  '    "BufferTooShort", "Condition", "Event", "JoinableQueue", "Lock",',
  '    "Manager", "Pipe", "Pool", "Process", "ProcessError", "Queue",',
  '    "RLock", "Semaphore", "SimpleQueue", "TimeoutError", "Value",',
  '    "active_children", "cpu_count", "current_process",',
  '    "get_all_start_methods", "get_context", "get_start_method",',
  '    "set_start_method"]',
  ''].join('\n'), true);
// spawn.py does `import multiprocessing.connection`
add('multiprocessing.connection', [
  'def wait(object_list, timeout=None):',
  '    return []',
  'def _browser_unavailable(*a, **k):',
  '    raise OSError("multiprocessing is not available in the browser")',
  'Connection = Client = Listener = Pipe = _browser_unavailable',
  ''].join('\n'), false);
// torch/multiprocessing/__init__.py imports ResourceTracker unguarded
// (the darwin leak workaround); hasattr(_RT, "__del__") must just be False
add('multiprocessing.resource_tracker', [
  'class ResourceTracker:',
  '    def register(self, name, rtype):',
  '        pass',
  '    def unregister(self, name, rtype):',
  '        pass',
  '_resource_tracker = ResourceTracker()',
  'def register(name, rtype):',
  '    pass',
  'def unregister(name, rtype):',
  '    pass',
  'def ensure_running():',
  '    pass',
  ''].join('\n'), false);
add('multiprocessing.reduction', [
  'class ForkingPickler:',
  '    def __init__(self, *a, **k):',
  '        raise OSError("multiprocessing is not available in the browser")',
  '    @classmethod',
  '    def register(cls, type, reduce):',
  '        pass',
  'def register(type, reduce):',
  '    pass',
  ''].join('\n'), false);
add('multiprocessing.util', [
  'def register_after_fork(obj, func):',
  '    pass',
  ''].join('\n'), false);

// torch._awaits: metaclass mix of pybind11 metatype + Generic (C3 clash
// bridge-side); Await support is out of the v1 slice
add('torch._awaits', [
  'class _Await:',
  '    pass',
  ''].join('\n'), true);

// distributed sub-stubs the kept scope probes
add('torch.distributed.rpc', [
  'def is_available():',
  '    return False',
  'def __getattr__(name):',
  '    raise ImportError("torch.distributed.rpc is not in this wasm build (v1)")',
  ''].join('\n'), false);

// spawn.py does `from . import _prctl_pr_set_pdeathsig`: in CPython that
// falls back to the package attribute the C _multiprocessing_init added;
// Brython's from-import has no attribute fallback, so serve a module (the
// name is only CALLED inside _wrap, which never runs in the browser)
add('torch.multiprocessing._prctl_pr_set_pdeathsig', [
  'def _prctl_pr_set_pdeathsig(sig):',
  '    pass',
  ''].join('\n'), false);

// torch/__init__ hard-imports cond/while_loop from the excluded HOP tree,
// and lazily probes _higher_order_ops.utils._in_hop_compile
add('torch._higher_order_ops', [
  'def cond(*a, **k):',
  '    raise ImportError("torch._higher_order_ops is not in this wasm build (v1)")',
  'def while_loop(*a, **k):',
  '    raise ImportError("torch._higher_order_ops is not in this wasm build (v1)")',
  'def __getattr__(name):',
  '    raise ImportError("torch._higher_order_ops is not in this wasm build (v1)")',
  ''].join('\n'), true);
// _prims/__init__ hard-imports new_token_tensor (only used when the
// effect-token prim is actually traced — never in the v1 slice)
add('torch._higher_order_ops.effects', [
  'def new_token_tensor(*a, **k):',
  '    raise ImportError("torch._higher_order_ops is not in this wasm build (v1)")',
  'def __getattr__(name):',
  '    raise ImportError("torch._higher_order_ops is not in this wasm build (v1)")',
  ''].join('\n'), false);
// _decomp/decompositions.py imports out_dtype at module level AND hands it
// to @register_decomposition, which requires a HigherOrderOperator (a
// plain function stub trips its assert); calling the hop still raises
add('torch._higher_order_ops.out_dtype', [
  'from torch._ops import HigherOrderOperator',
  'class _OutDtypeStub(HigherOrderOperator):',
  '    def __init__(self):',
  '        super().__init__("out_dtype")',
  '    def __call__(self, *a, **k):',
  '        raise ImportError("torch._higher_order_ops is not in this wasm build (v1)")',
  'out_dtype = _OutDtypeStub()',
  'def __getattr__(name):',
  '    raise ImportError("torch._higher_order_ops is not in this wasm build (v1)")',
  ''].join('\n'), false);
// torch/compiler/__init__ hard-imports this options dataclass (only ever
// INSTANTIATED by compile paths, dead in v1)
add('torch._higher_order_ops.invoke_subgraph', [
  'class NestedCompileRegionOptions:',
  '    def __init__(self, fw_compiler=None, bw_compiler=None, *a, **k):',
  '        self.fw_compiler = fw_compiler',
  '        self.bw_compiler = bw_compiler',
  'def __getattr__(name):',
  '    raise ImportError("torch._higher_order_ops is not in this wasm build (v1)")',
  ''].join('\n'), false);
// nn/attention/flex_attention.py imports the HOP pair at module level; a
// HigherOrderOperator subclass satisfies registration, calling still raises
// (same shape as the out_dtype stub)
add('torch._higher_order_ops.flex_attention', [
  'from torch._ops import HigherOrderOperator',
  'class _FlexStub(HigherOrderOperator):',
  '    def __call__(self, *a, **k):',
  '        raise ImportError("torch._higher_order_ops is not in this wasm build (v1)")',
  'flex_attention = _FlexStub("flex_attention")',
  'flex_attention_backward = _FlexStub("flex_attention_backward")',
  'def __getattr__(name):',
  '    raise ImportError("torch._higher_order_ops is not in this wasm build (v1): attr " + name)',
  ''].join('\n'), false);
add('torch._higher_order_ops.utils', [
  'def _in_hop_compile(*a, **k):',
  '    return False',
  // rng_prims imports these at module level; they only DECORATE hop
  // registrations, so pass-through/None keeps the definitions inert
  'def autograd_not_implemented(op, deferred_error=False):',
  '    def _fn(*a, **k):',
  '        raise NotImplementedError("autograd not implemented (wasm v1)")',
  '    return _fn',
  'def register_fake(op, *a, **k):',
  '    def _deco(fn):',
  '        return fn',
  '    return _deco',
  'def __getattr__(name):',
  '    raise ImportError("torch._higher_order_ops is not in this wasm build (v1)")',
  ''].join('\n'), false);

// torch.testing._internal ships real (walked above) — the former stubs
// (package init, common_dtype, logging_tensor) are the genuine modules now.

// jit/_builtins.py does `import torch.distributed.autograd as dist_autograd`
// then guards on dist_autograd.is_available() — same shape as rpc
add('torch.distributed.autograd', [
  'def is_available():',
  '    return False',
  'def __getattr__(name):',
  '    raise ImportError("torch.distributed.autograd is not in this wasm build (v1)")',
  ''].join('\n'), false);

// nn/parallel/distributed.py hard-imports Join/Joinable/JoinHook at module
// level and DistributedDataParallel INHERITS Joinable, so these must be
// real classes; they only raise if actually used (dist.is_available() is
// False so no DDP path ever runs in v1)
add('torch.distributed.algorithms', '', true);
add('torch.distributed.algorithms.join', [
  'class JoinHook:',
  '    def main_hook(self):',
  '        pass',
  '    def post_hook(self, is_last_joiner):',
  '        pass',
  'class Joinable:',
  '    pass',
  'class Join:',
  '    def __init__(self, *a, **k):',
  '        raise ImportError("torch.distributed is not in this wasm build (v1)")',
  ''].join('\n'), false);

// torch._C._dynamo: the C dynamo bindings are excluded (eval-frame
// internals); serve the few names the kept scope imports as no-ops
add('torch._C._dynamo', '', true);
add('torch._C._dynamo.guards', [
  'def set_is_in_mode_without_ignore_compile_internals(*a, **k):',
  '    pass',
  'def get_is_in_mode_without_ignore_compile_internals(*a, **k):',
  '    return False',
  'def __getattr__(name):',
  '    raise ImportError("torch._C._dynamo.guards is not in this wasm build (v1): " + name)',
  ''].join('\n'), false);
add('torch._C._dynamo.eval_frame', [
  'def __getattr__(name):',
  '    raise ImportError("torch._C._dynamo.eval_frame is not in this wasm build (v1): " + name)',
  ''].join('\n'), false);

// pickletools: pure stdlib module Brython lacks (torch.package uses it)
add('pickletools', fs.readFileSync(path.join(HERE, 'vendor', 'pickletools.py'), 'utf8'), false);

// typing_extensions from the host python (torch imports it everywhere)
add('typing_extensions',
    fs.readFileSync(path.join(HERE, 'vendor', 'typing_extensions.py'), 'utf8'),
    false);

// common_utils does `import __main__` (guarded getattr(__main__, '__file__')
// uses only); Brython only registers __main__ for an unnamed inline script,
// and the dashboard drives runPythonSource with explicit names
add('__main__', '# browser: no main script\n', false);

// runpy (vendored CPython, pure): common_device_type imports it at module
// level; Brython's stdlib bundle does not serve it
add('runpy',
    fs.readFileSync(path.join(HERE, 'vendor', 'runpy.py'), 'utf8'),
    false);

// common_device_type imports GPU_TYPES from the excluded inductor tree
// (module level); value replicated from torch/_inductor/utils.py:107
add('torch._inductor.utils', [
  'GPU_TYPES = ["cuda", "mps", "xpu", "mtia"]',
  'def __getattr__(name):',
  '    raise ImportError("torch._inductor is not in this wasm build (v1)")',
  ''].join('\n'), false);

// logging_utils imports LazyString (a deferred-format helper, pure python
// semantics replicated) from the excluded dynamo tree
add('torch._dynamo.utils', [
  'class LazyString:',
  '    def __init__(self, func, *args, **kwargs):',
  '        self.func = func',
  '        self.args = args',
  '        self.kwargs = kwargs',
  '    def __str__(self):',
  '        return self.func(*self.args, **self.kwargs)',
  'def __getattr__(name):',
  '    raise ImportError("torch._dynamo is not in this wasm build (v1): attr utils." + name)',
  ''].join('\n'), false);

// autograd/test_logging.py mentions it under `if __name__ == "__main__"` —
// Brython statically pre-resolves every import in the source, so the module
// must exist; the eager-build equivalents are the common_utils ones
add('torch._dynamo.test_case', [
  'from torch.testing._internal.common_utils import TestCase, run_tests',
  ''].join('\n'), false);

add('torch._export.utils', [
  'def __getattr__(name):',
  '    raise ImportError("torch._export is not in this wasm build (v1): attr utils." + name)',
  ''].join('\n'), false);

// two_tensor.py (test_serialization) imports this experimental export marker
// at module level; it only tags constructors for the excluded export tracer
add('torch._export.wrappers', [
  'def mark_subclass_constructor_exportable_experimental(fn):',
  '    return fn',
  'def __getattr__(name):',
  '    raise ImportError("torch._export is not in this wasm build (v1)")',
  ''].join('\n'), false);

// numpy.fft OVERRIDE (torch_vfs loads after numpy_vfs, so this wins):
// NumBry's numpy wasm carries no pocketfft C module, and the real
// numpy/fft/__init__ import dies half-way — while the OpInfo tables
// (opinfo/definitions/fft.py) reference np.fft.* AT IMPORT TIME. Serve
// callables that raise at CALL, so the suites import and fft-referencing
// tests fail/skip honestly.
add('numpy.fft', [
  'def _missing(_name):',
  '    def _fn(*a, **k):',
  '        raise NotImplementedError("numpy.fft." + _name + " is not in this browser numpy build")',
  '    _fn.__name__ = _name',
  '    return _fn',
  '',
  'for _name in ("fft", "ifft", "fft2", "ifft2", "fftn", "ifftn",',
  '              "rfft", "irfft", "rfft2", "irfft2", "rfftn", "irfftn",',
  '              "hfft", "ihfft", "fftshift", "ifftshift", "fftfreq",',
  '              "rfftfreq"):',
  '    globals()[_name] = _missing(_name)',
  ''].join('\n'), true);

// expecttest (vendored, MIT): common_utils hard-imports it
add('expecttest',
    fs.readFileSync(path.join(HERE, 'vendor', 'expecttest.py'), 'utf8'),
    false);

// The official pytorch suites the dashboard runs, embedded as top-level
// modules (pytorch CI runs them the same way: `cd test && python test_X.py`).
const TEST_SUITES = [
  'test_testing', 'test_indexing', 'test_view_ops', 'test_shape_ops',
  'test_type_promotion', 'test_sort_and_select', 'test_reductions',
  'test_serialization', 'test_autograd', 'test_torch',
];
for (const t of TEST_SUITES) {
  add(t, fs.readFileSync(path.join(PT, 'test', t + '.py'), 'utf8'), false);
}
// test_autograd imports its sibling package (`from autograd.test_complex
// import …`); upstream runs with test/ as cwd, we serve it as a VFS package
add('autograd', '', true);
for (const f of fs.readdirSync(path.join(PT, 'test', 'autograd'))) {
  if (!f.endsWith('.py')) continue;
  add('autograd.' + f.replace(/\.py$/, ''),
      fs.readFileSync(path.join(PT, 'test', 'autograd', f), 'utf8'), false);
}

const blob = ';(function(){\nif(typeof __BRYTHON__==="undefined"){throw new Error("load brython.js first")}\n__BRYTHON__.update_VFS(' + JSON.stringify(scripts) + ');\n})();\n';
fs.mkdirSync(path.join(ROOT, 'build'), { recursive: true });
fs.writeFileSync(OUT, blob);
console.log('torch VFS: ' + n + ' modules, ' + (bytes / 1048576).toFixed(1) + ' MB src, blob ' + (blob.length / 1048576).toFixed(1) + ' MB -> ' + OUT);
