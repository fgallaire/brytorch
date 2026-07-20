#!/usr/bin/env bash
# BryTorch — build the REAL PyTorch (torch._C, whole-archive ATen) as a
# WebAssembly module + torch's pure-Python layer as a Brython VFS. Artifacts
# are NEVER committed: they are produced here (locally or in CI) from source.
#
# The generic C-API bridge (src/wasthon.*) and Brython come from the wasthon
# repo (@main); the conversion tool, the compat layer, the VFS generator and
# the loader page live HERE. PyTorch is pinned to the exact commit the port
# was validated against.
#
# Stages (default: all):
#   deps       emsdk + cmake/ninja + host protoc
#   sources    clone wasthon + pytorch@pin (+submodules), apply the port
#   aten       cmake configure + ninja c10 torch_cpu   (LONG: hours on 4 vCPU)
#   pack-aten  tar the build tree (libs + generated headers) for CI reuse
#   bindings   compile the ~150 torch_python binding TUs against wasthon.h
#   link       emcc link -> build/npth.mjs + npth.wasm
#   vfs        build/torch_vfs.js (torch's Python layer as a Brython VFS)
#   site       split the wasm into <95 MB parts + stage loader/brython
#   ci         deps sources unpack-aten bindings link vfs site
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

WASTHON_REPO="${WASTHON_REPO:-https://github.com/fgallaire/wasthon.git}"
WASTHON_REF="${WASTHON_REF:-main}"
EMSCRIPTEN_VERSION="${EMSCRIPTEN_VERSION:-5.0.7}"
PYTORCH_REPO="${PYTORCH_REPO:-https://github.com/pytorch/pytorch.git}"
PYTORCH_PIN="d0458e4559f6f27b3c18a35ab3697956f6145174"   # 2.14.0a0, port calibrated on it
PROTOC_VERSION="21.12"                                    # host protoc (wasm protoc can't run)
ATEN_TARBALL="aten-wasm-${PYTORCH_PIN:0:7}.tar.zst"

W="$HERE/.wasthon"
PT="$HERE/pytorch"
BW="$PT/build-wasm"
OBJ="$BW/bindings-obj"
mkdir -p "$HERE/build"

CMAKE_FLAGS=(-GNinja -DCMAKE_BUILD_TYPE=Release
  -DPython_EXECUTABLE="$(command -v python3)"
  -DONNX_CUSTOM_PROTOC_EXECUTABLE="$HERE/.protoc/bin/protoc"
  -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE="$HERE/.protoc/bin/protoc"
  -DCMAKE_CXX_FLAGS= -DCMAKE_C_FLAGS=
  -DBUILD_PYTHON=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_TEST=OFF
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_XPU=OFF -DUSE_MPS=OFF -DUSE_DISTRIBUTED=OFF
  -DUSE_MKLDNN=OFF -DUSE_FBGEMM=OFF -DUSE_XNNPACK=OFF -DUSE_NNPACK=OFF
  -DUSE_PYTORCH_QNNPACK=OFF -DUSE_OPENMP=OFF -DUSE_KINETO=OFF -DUSE_NUMPY=OFF
  -DUSE_ITT=OFF -DUSE_NCCL=OFF -DUSE_GLOO=OFF -DUSE_TENSORPIPE=OFF -DUSE_CUDNN=OFF
  -DUSE_VULKAN=OFF -DUSE_OPENCL=OFF -DUSE_BLAS=OFF -DUSE_LAPACK=OFF
  -DUSE_MAGMA=OFF -DUSE_METAL=OFF)

# Compile line shared by every binding TU. JS-emulated exceptions EVERYWHERE
# (-s DISABLE_EXCEPTION_CATCHING=0): the CMake side of the build uses them,
# and -fwasm-exceptions does not mix.
BINDING_FLAGS=(-std=gnu++20 -O2 -s DISABLE_EXCEPTION_CATCHING=0
  -DONNX_ML=1 -DONNX_NAMESPACE=onnx_torch
  -include "$HERE/src/compat/torch_compat.h"
  -I "$W/src" -I "$HERE/src/compat"
  -I "$PT" -I "$PT/aten/src" -I "$PT/c10" -I "$BW" -I "$BW/aten/src"
  -I "$PT/third_party/pybind11/include" -I "$PT/third_party/fmt/include"
  -I "$PT/torch/csrc/api/include" -I "$PT/torch/lib/libshm"
  -I "$PT/third_party/nlohmann/include"
  -I "$PT/third_party/kineto/libkineto/include"
  -I "$PT/third_party/onnx" -I "$BW/third_party/onnx"
  -I "$PT/third_party/protobuf/src")

need_emsdk() { source "$HERE/.emsdk/emsdk_env.sh" >/dev/null 2>&1; }

stage_deps() {
  echo "=== deps: emsdk ${EMSCRIPTEN_VERSION} + cmake/ninja + protoc ${PROTOC_VERSION} ==="
  if [ ! -d "$HERE/.emsdk" ]; then
    git clone --depth 1 https://github.com/emscripten-core/emsdk.git "$HERE/.emsdk"
    ( cd "$HERE/.emsdk" && ./emsdk install "$EMSCRIPTEN_VERSION" && ./emsdk activate "$EMSCRIPTEN_VERSION" )
  fi
  python3 -m pip install --user --quiet --break-system-packages cmake ninja pyyaml typing_extensions packaging 2>/dev/null \
    || python3 -m pip install --user --quiet cmake ninja pyyaml typing_extensions packaging
  if [ ! -x "$HERE/.protoc/bin/protoc" ]; then
    mkdir -p "$HERE/.protoc"
    curl -fsSL -o /tmp/protoc.zip \
      "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip"
    ( cd "$HERE/.protoc" && python3 -c "import zipfile;zipfile.ZipFile('/tmp/protoc.zip').extractall()" && chmod +x bin/protoc )
  fi
}

stage_sources() {
  echo "=== sources: wasthon @ ${WASTHON_REF} ==="
  rm -rf "$W"; git clone --depth 1 -b "$WASTHON_REF" "$WASTHON_REPO" "$W"
  echo "=== sources: pytorch @ ${PYTORCH_PIN} ==="
  if [ ! -d "$PT/.git" ]; then
    git init -q "$PT"
    git -C "$PT" remote add origin "$PYTORCH_REPO" 2>/dev/null || true
    git -C "$PT" fetch --depth 1 origin "$PYTORCH_PIN"
    git -C "$PT" checkout -q FETCH_HEAD
  fi
  # PyTorch's cmake/PreBuildSteps.cmake verifies that EVERY top-level
  # submodule listed in .gitmodules is populated — a curated partial init
  # trips it (e.g. on third_party/googletest) and it only self-heals when
  # *nothing* is initialized. So init them all (shallow), plus the one
  # nested submodule the build actually reaches (fbgemm's asmjit).
  git -C "$PT" submodule update --init --depth 1
  git -C "$PT/third_party/fbgemm" submodule update --init --depth 1 external/asmjit 2>/dev/null || true
  echo "=== sources: apply the port ==="
  # 1) the three recette-patches (the ONLY C++ edits in 1.1M lines)
  python3 - "$PT" << 'PYEOF'
import sys, pathlib
pt = pathlib.Path(sys.argv[1])
f = pt / 'torch/csrc/autograd/python_cpp_function.cpp'
s = f.read_text()
if 'type.ob_base' in s:
    import re
    s = re.sub(r'type\.ob_base\s*=\s*\{[^}]*\};', 'type.ob_refcnt = 0;  /* brytorch: wasthon PyObject layout */', s)
    f.write_text(s)
f = pt / 'torch/csrc/autograd/python_engine.cpp'
s = f.read_text()
old = '      !PyGILState_Check(),'
if old in s:
    # single-threaded wasm: the engine's anti-deadlock "GIL must NOT be
    # held" guard is moot (the bridge's PyGILState_Check is constant 1,
    # and every other site asserts the OPPOSITE polarity)
    s = s.replace(old, '      true,  /* brytorch: single-thread, anti-deadlock GIL guard moot */')
    f.write_text(s)
f = pt / 'torch/csrc/utils/pybind.h'
s = f.read_text()
old = 'PYBIND11_DECLARE_HOLDER_TYPE(T, c10::SingletonOrSharedTypePtr<T>)'
if old in s:
    s = s.replace(old,
        '// brytorch: `true` = constructible from a raw T* (the c10 ctor exists\n'
        '// "for pybind"); without it the first Python instance of an interned jit\n'
        '// Type can be cached holder-less and every later holder load throws.\n'
        'PYBIND11_DECLARE_HOLDER_TYPE(T, c10::SingletonOrSharedTypePtr<T>, true)')
    f.write_text(s)
PYEOF
  # 2) 31 static PyTypeObject initializers -> designated, against wasthon.h
  ( cd "$PT" && python3 "$HERE/src/torchconvert.py" "$W/src/wasthon.h" \
      $(cat "$HERE/src/converted_files.txt") )
}

stage_aten() {
  need_emsdk
  echo "=== aten: emcmake configure ==="
  mkdir -p "$BW"
  ( cd "$BW" && PATH="$HOME/.local/bin:$PATH" emcmake cmake .. "${CMAKE_FLAGS[@]}" )
  echo "=== aten: ninja c10 torch_cpu (the long part) ==="
  ( cd "$BW" && PATH="$HOME/.local/bin:$PATH" ninja c10 torch_cpu )
}

stage_pack_aten() {
  echo "=== pack-aten: ${ATEN_TARBALL} (libs + generated headers, no objects) ==="
  ( cd "$HERE" && tar --zstd -cf "build/${ATEN_TARBALL}" \
      --exclude='*.o' --exclude='CMakeFiles' --exclude='.ninja_*' \
      pytorch/build-wasm/lib \
      pytorch/build-wasm/aten pytorch/build-wasm/c10 pytorch/build-wasm/caffe2 \
      pytorch/build-wasm/third_party/onnx \
      pytorch/torch/csrc/autograd/generated \
      pytorch/torch/csrc/functionalization/generated \
      pytorch/torch/csrc/inductor 2>/dev/null || true )
  ls -la "$HERE/build/${ATEN_TARBALL}"
}

stage_unpack_aten() {
  echo "=== unpack-aten: ${ATEN_TARBALL} ==="
  tar --zstd -xf "$HERE/build/${ATEN_TARBALL}" -C "$HERE"
}

stage_bindings() {
  need_emsdk
  echo "=== bindings: ~150 torch_python TUs against wasthon.h ==="
  mkdir -p "$OBJ"
  compile_obj() {
    local f="$1"
    local o="$OBJ/$(echo "$f" | tr / _ | sed 's/\.cpp$//;s/\.c$//').o"
    [ -f "$o" ] && return 0
    em++ "${BINDING_FLAGS[@]}" -c "$PT/$f" -o "$o" 2> "${o%.o}.err" || { echo "FAIL $f"; cat "${o%.o}.err" | head -5; return 1; }
  }
  export -f compile_obj 2>/dev/null || true
  local fails=0
  while read -r f; do compile_obj "$f" &
    while [ "$(jobs -r | wc -l)" -ge "$(nproc)" ]; do wait -n || fails=$((fails+1)); done
  done < "$HERE/src/binding_sources.txt"
  wait || true
  for f in torch/csrc/onnx/init.cpp torch/csrc/jit/passes/onnx.cpp \
           torch/csrc/jit/passes/onnx/shape_type_inference.cpp \
           torch/csrc/jit/passes/onnx/helper.cpp torch/csrc/jit/passes/onnx/constant_map.cpp \
           torch/csrc/functionalization/generated/ViewMetaClassesPythonBinding.cpp \
           torch/csrc/lazy/python/init.cpp torch/csrc/lazy/python/python_util.cpp; do
    compile_obj "$f"
  done
  em++ "${BINDING_FLAGS[@]}" -c "$HERE/src/compat/torch_stubs.cpp" -o "$OBJ/torch_stubs.o"
  em++ -std=gnu++20 -O2 -s DISABLE_EXCEPTION_CATCHING=0 -c "$PT/third_party/fmt/src/format.cc" \
       -I "$PT/third_party/fmt/include" -o "$OBJ/fmt_format.o"
  emcc -O2 -c "$HERE/src/compat/torch_compat.c" -I "$W/src" -I "$HERE/src/compat" -o "$OBJ/torch_compat.o"
  emcc -O2 -c "$W/src/wasthon.c" -I "$W/src" -o "$OBJ/wasthon.o"
  emcc -O2 -c "$PT/torch/csrc/stub.c" -I "$W/src" -o "$OBJ/stub.o"
  echo "bindings objects: $(ls "$OBJ"/*.o | wc -l)"
}

stage_link() {
  need_emsdk
  echo "=== link: build/npth.mjs + npth.wasm ==="
  emcc -O2 -s DISABLE_EXCEPTION_CATCHING=0 "$OBJ"/*.o \
    --js-library "$W/src/wasthon.js" \
    -Wl,--whole-archive "$BW/lib/libtorch_cpu.a" -Wl,--no-whole-archive \
    "$BW/lib/libc10.a" "$BW/lib/libcpuinfo.a" \
    "$BW/lib/libonnx.a" "$BW/lib/libonnx_proto.a" "$BW/lib/libprotobuf.a" \
    -sFORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=67108864 -sMAXIMUM_MEMORY=2147483648 -s ALLOW_TABLE_GROWTH=1 -sSTACK_SIZE=5242880 \
    -s EXPORTED_FUNCTIONS='["_PyInit__C","_wasthon_init","_wasthon_module_create","_wasthon_torch_property_addr","_wasthon_torch_staticmethod_addr","_malloc","_free"]' \
    -s EXPORTED_RUNTIME_METHODS='["HEAPU8","HEAP32","HEAPF32","HEAPF64","HEAP16","UTF8ToString","stringToUTF8","lengthBytesUTF8","addFunction","FS"]' \
    -s MODULARIZE=1 -s EXPORT_ES6=1 -s EXPORT_NAME=createTorchModule \
    -o "$HERE/build/npth.mjs"
  # Size-optimize the linked wasm: a POST-LINK wasm-opt -Oz harvests the
  # cross-module dedup + dead-code elimination the -O2 link leaves on the
  # table for speed (ATen is heavily templated → many near-identical
  # instantiations). ~30% smaller (121 -> 86 MB), which keeps it under the
  # GitHub Pages per-file ceiling so it ships as ONE file. Modest potential
  # cost to hot-kernel speed, irrelevant for the eager v1.
  echo "=== link: wasm-opt -Oz (size) ==="
  "$HERE/.emsdk/upstream/bin/wasm-opt" --all-features -Oz \
    -o "$HERE/build/npth.opt.wasm" "$HERE/build/npth.wasm"
  mv "$HERE/build/npth.opt.wasm" "$HERE/build/npth.wasm"
  ls -la "$HERE/build/npth.wasm"
}

stage_vfs() {
  echo "=== vfs: build/torch_vfs.js ==="
  node "$HERE/src/gen_torch_vfs.mjs"
}

# NumBry (numpy-on-Brython, same bridge family) published artifacts — the
# official-suite dashboard loads the real numpy wasm alongside torch so the
# numpy-referencing pytorch tests run against real arrays. Fetched, never
# built here (NumBry's own CI builds them).
NUMBRY_SITE="${NUMBRY_SITE:-https://fgallaire.github.io/numbry}"
stage_numbry() {
  echo "=== numbry: fetch published NumBry artifacts ==="
  for f in nprnd.mjs nprnd.wasm numpy_vfs.js; do
    curl -fsSL -o "$HERE/build/$f" "$NUMBRY_SITE/build/$f"
  done
  ls -la "$HERE/build/nprnd.mjs" "$HERE/build/nprnd.wasm" "$HERE/build/numpy_vfs.js"
}

stage_site() {
  echo "=== site: CPython suite dashboard (wasthon-full bundle, from the wasthon clone) ==="
  ( cd "$W" && bash build.sh wasthon-full )
  cp "$W/build/wasthon-full.mjs" "$W/build/wasthon-full.wasm" "$HERE/build/"
  cp "$W"/loader/test-cpython-all.html "$W"/loader/test-cpython.html \
     "$W"/loader/brython-src.js "$W"/loader/wasthon-loader.js \
     "$W"/loader/wasthon-io-write.js "$W"/loader/wasthon-fs.js \
     "$W"/loader/wasthon-dealloc.js "$W"/loader/wasthon-dbm.js "$HERE/loader/"
  rm -rf "$HERE/loader/cpython-tests"
  cp -r "$W/loader/cpython-tests" "$HERE/loader/"
  echo "=== site: single wasm if it fits, else split into parts + stage brython ==="
  python3 - "$HERE/build" << 'PYEOF'
import json, os, sys
b = sys.argv[1]
src = os.path.join(b, 'npth.wasm')
size = os.path.getsize(src)
LIMIT = 95 * 1024 * 1024   # GitHub Pages per-file ceiling (~100 MB), with margin
# clear any stale split artifacts from a previous build
for f in os.listdir(b):
    if f.startswith('npth.wasm.part') or f == 'npth.parts.json':
        os.remove(os.path.join(b, f))
if size <= LIMIT:
    print(f'{size} bytes -> single file (fits under {LIMIT})')
else:
    CHUNK = 90 * 1024 * 1024
    parts = []
    with open(src, 'rb') as f:
        i = 0
        while True:
            data = f.read(CHUNK)
            if not data: break
            name = f'npth.wasm.part{i:02d}'
            open(os.path.join(b, name), 'wb').write(data)
            parts.append(name); i += 1
    json.dump({'total': size, 'parts': parts}, open(os.path.join(b, 'npth.parts.json'), 'w'))
    print(f'{size} bytes -> {len(parts)} part(s)')
PYEOF
  rm -rf "$HERE/loader/brython"
  cp -r "$W/loader/brython" "$HERE/loader/brython"
}

run() { for s in "$@"; do "stage_${s//-/_}"; done }
case "${1:-all}" in
  all) run deps sources aten pack-aten bindings link vfs numbry site ;;
  ci)  run deps sources unpack-aten bindings link vfs numbry site ;;
  *)   run "$@" ;;
esac
echo "=== done ==="
