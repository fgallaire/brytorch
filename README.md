# BryTorch

**The real PyTorch running in the browser** — the genuine `torch._C`
(whole-archive ATen, 1.1M lines of C++) compiled to WebAssembly, plus
torch's own pure-Python layer running on
[Brython](https://github.com/brython-dev/brython) through the
[Wasthon](https://github.com/fgallaire/wasthon) C-API bridge.

Not a reimplementation, not a transpilation: upstream PyTorch itself.

```pycon
>>> import torch
>>> t = torch.tensor([1, 2])
>>> t + 1
tensor([2, 3])
```

`import torch` traverses torch's own `__init__` end to end — `torch._C`
initializes its 24 C submodules and 926 attributes, `torch.Tensor` is the
real `THPVariable` (and `nn.Parameter` a Python subclass of it through the
C metatype), the dtypes, `nn`, `autograd`, `fx`, `optim`, the ~300 prim
registrations and the full decomposition table all load — and the math is
executed by the real ATen kernels in wasm.

## How it works

- **ATen + c10 compile untouched**: `emcmake cmake` + `ninja torch_cpu`
  build the entire tensor library to wasm with **zero source changes**
  (CPU-only, single-thread, every `USE_*` off). The kernel registries are
  static initializers, so the link is `--whole-archive`.
- **The Python binding layer** (~150 `torch_python` TUs: pybind11 3.0 +
  hand-written CPython C-API) is compiled against **wasthon.h**, the
  Wasthon bridge's CPython-compatible header. A conversion tool
  (`src/torchconvert.py`) rewrites the 31 static `PyTypeObject`
  initializers (1029 positional slots) to designated initializers; a small
  compat layer (`src/compat/`) supplies the C-API surface the bridge keeps
  JS-side. **Two one-line C++ patches** in 1.1M lines
  (see `build.sh`, "recette-patches").
- **torch's Python layer** (~1270 modules) is served as a Brython VFS
  (`src/gen_torch_vfs.mjs`), with the compile/export infrastructure
  (`_dynamo`, `_inductor`, `torch.export`, distributed, onnx-python)
  stubbed out of the v1 slice — the same subtrees other browser Pythons
  exclude.

## v1 scope

Eager mode: tensor creation, arithmetic, matmul, reductions, indexing,
`repr` — everything the real dispatcher + ATen CPU kernels provide.
Excluded for now: `torch.compile`/export/fx-tracing paths, distributed,
multiprocessing (no fork in wasm), CUDA (obviously).

### Where the edge actually is

51 upstream suites were probed for importability; **30 import as they stand**.
Of the 21 that do not, most are held up by a missing pure-Python package or a
stub still to write — `packaging`, `setuptools`, `yaml`, `functorch`, a
sibling test package, `_inductor.async_compile`. Only two causes are
structural: `optree`, a C extension, and `_dynamo`, which needs CPython's
frame-evaluation API and cannot exist here. `torch.export` sits behind
`_dynamo` — it imports it at module level, though only for exception classes
and decorators that are inert without a compiler.

So the boundary is not where the v1 slice drew it. Most of what is missing is
plumbing; what is genuinely out of reach is the compiler.

## Measured against PyTorch's own test suites

The dashboards run upstream `pytorch/test/test_*.py` **unmodified**, with the
real `torch.testing._internal` harness, against the wasm `torch._C`. Nothing
is rewritten to pass: a suite says what it says.

- **1/3** — `test_torch`, then the quick suites
- **2/3** — `test_autograd`, then `test_reductions`: 4684 tests, 30 failures,
  none of them caused by memory
- **3/3** — a second tier of nineteen further suites in one frame,
  1434/1465 pass; eleven of them green the first time they were ever run

A suite gets its own iframe when it needs one, and no sooner. A wasm heap
only ever grows, and destroying a frame is the only real free there is — so
frames are how a suite that wants a gigabyte stops being the reason the next
one fails.

## Build

Everything is built from source; no artifacts are committed.

```sh
./build.sh          # full build: emsdk + pytorch@pin + ATen + bindings + link + VFS
./build.sh ci       # what CI runs: reuses the ATen libs from the Release asset
```

The ATen half takes hours (it is a full libtorch compile); the torch-python
half takes ~30 min. CI therefore consumes a prebuilt
`aten-wasm-<pin>.tar.zst` Release asset (libs + generated headers, no
objects) produced once by `./build.sh aten pack-aten` — locally or via the
manual `rebuild-aten` workflow — and rebuilds all the Python-facing parts
from source on every push.

A post-link `wasm-opt -Oz` pass shrinks the linked module ~30% (harvesting
the cross-module dedup + dead-code the speed-first `-O2` link leaves behind,
at build-time cost only), keeping it under GitHub Pages' ~100 MB per-file
ceiling so it ships as a single file. Should a future build exceed the
ceiling, `build.sh` splits it into <95 MiB parts + a manifest that the
loader reassembles before instantiation.

## License

Copyright (C) 2026 Florent Gallaire <fgallaire@gmail.com>

BSD 3-Clause License — same as Brython. See `LICENSE` for the full text.
