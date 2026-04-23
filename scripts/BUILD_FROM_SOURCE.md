# Building OpenVINO + OpenVINO GenAI from Source on Ubuntu

All commands assume you are working from `/mnt/Ironwolf-4TB/openvino-hetero-test/`.
Tested on Ubuntu 24.04, gcc 13.3, 4 cores / 15 GB RAM (sandbox), against OpenVINO
master `74c0aa5e` and openvino.genai master.

## 1. Clone both repos

```sh
cd /mnt/Ironwolf-4TB/openvino-hetero-test
git clone --recursive https://github.com/openvinotoolkit/openvino.git
git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
```

## 2. Install system build dependencies

```sh
cd /mnt/Ironwolf-4TB/openvino-hetero-test/openvino
sudo ./install_build_dependencies.sh
cd /mnt/Ironwolf-4TB/openvino-hetero-test
```

## 3. Create venv with uv (use --seed so pip is included)

```sh
uv venv /mnt/Ironwolf-4TB/openvino-hetero-test/.venv --python 3.12 --seed
source /mnt/Ironwolf-4TB/openvino-hetero-test/.venv/bin/activate
```

## 4. Install build dependencies

```sh
uv pip install -r /mnt/Ironwolf-4TB/openvino-hetero-test/openvino/src/bindings/python/wheel/requirements-dev.txt
uv pip install numpy pybind11-stubgen==2.5.5
```

Do **not** pin `py-build-cmake` here. We install it per-subpackage below because
openvino_tokenizers and openvino_genai need different versions.

## 5. Configure OpenVINO with CMake

```sh
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_PYTHON=ON \
  -DENABLE_WHEEL=ON \
  -DPython3_EXECUTABLE=$(which python3) \
  -S /mnt/Ironwolf-4TB/openvino-hetero-test/openvino \
  -B /mnt/Ironwolf-4TB/openvino-hetero-test/openvino/build
```

## 6. Build OpenVINO wheel

**Target is `ie_wheel`, not `wheel`.** On OpenVINO 2026 master there is no
top-level `wheel` target — you get `gmake: *** No rule to make target 'wheel'`.

```sh
cmake --build /mnt/Ironwolf-4TB/openvino-hetero-test/openvino/build --parallel $(nproc) --target ie_wheel
```

Takes ~50 min on 4 cores. Produces `build/wheels/openvino-*cp312*.whl`.

## 7. Install the OpenVINO wheel

```sh
uv pip install --force-reinstall --no-deps \
  /mnt/Ironwolf-4TB/openvino-hetero-test/openvino/build/wheels/openvino-*cp312-cp312-*.whl
```

## 8. Build and install openvino_tokenizers wheel

Install the version of `py-build-cmake` that tokenizers pins. **Do not use
0.5.0** — it rejects the upstream `thirdparty/openvino_tokenizers/pyproject.toml`
with `"project.license-files" must not be used when "project.license" is not
a SPDX license expression` (PEP 639 strict validation). 0.4.3 is what the
tokenizers pyproject.toml itself pins.

```sh
uv pip install 'py-build-cmake==0.4.3'
```

Wipe any stale py-build-cmake cache before building — it caches `OpenVINO_DIR`
and will silently point at a different venv, which shows up as a bogus
`openvino/frontend/tensorflow/extension/conversion.hpp: No such file or directory`
compile error:

```sh
rm -rf /mnt/Ironwolf-4TB/openvino-hetero-test/openvino.genai/thirdparty/openvino_tokenizers/.py-build-cmake_cache
rm -rf /tmp/tokenizers_wheel
```

Build:

```sh
cd /mnt/Ironwolf-4TB/openvino-hetero-test/openvino.genai/thirdparty/openvino_tokenizers
OpenVINO_DIR=/mnt/Ironwolf-4TB/openvino-hetero-test/.venv/lib/python3.12/site-packages/openvino/cmake \
  pip wheel . --no-deps --no-build-isolation \
  --wheel-dir /tmp/tokenizers_wheel
cd /mnt/Ironwolf-4TB/openvino-hetero-test

uv pip install --force-reinstall --no-deps \
  /tmp/tokenizers_wheel/openvino_tokenizers-*.whl
```

## 9. Build and install openvino_genai wheel

openvino.genai's own pyproject.toml is SPDX-clean and pins `py-build-cmake==0.5.0`.
Upgrade:

```sh
uv pip install 'py-build-cmake==0.5.0'
```

Wipe its cache too:

```sh
rm -rf /mnt/Ironwolf-4TB/openvino-hetero-test/openvino.genai/.py-build-cmake_cache
rm -rf /tmp/genai_wheel
```

### 9a. Patch the `gguf_utils/format` template (required on gcc 13)

Before building, apply a two-file patch to fix an upstream template-linkage
bug that only manifests on gcc 13. On gcc 11 (Ubuntu 22.04 CI) and gcc 14
with `-fno-lto` (manylinux_2_28 CI) the bug doesn't trigger, which is why
it survives in master.

**The bug.** `src/cpp/src/gguf_utils/gguf.hpp` declares a variadic template
`format` at global scope, and `gguf.cpp` provides the definition *in the
.cpp file*. Any other TU that includes `gguf.hpp` (e.g. `building_blocks.cpp`)
sees only the declaration, so the compiler emits an external reference to
`format<int>`. gcc 13 at `-O3 -fvisibility-inlines-hidden` then ISRA-clones
the `gguf.cpp`-local instantiations into a local-only specialization
(`format<int>.isra.0`) and drops the normal external symbol entirely, so
`building_blocks.o`'s reference has nothing to resolve against. The linker
permits it (shared libs default to allowing unresolved symbols), and `dlopen`
fails at wheel-build stub-generation time with:

```
ImportError: .../libopenvino_genai.so.2620:
  undefined symbol: _Z6formatIJiEENSt7__cxx1112basic_stringI...
```

**The fix.** Move the template body from `gguf.cpp` into `gguf.hpp` and mark
it `inline`. Every TU that includes the header then sees the full definition
and can instantiate it locally — no cross-TU symbol dependency is possible.

**Edit 1: `src/cpp/src/gguf_utils/gguf.hpp`, lines 30-31.** Replace this
declaration:

```cpp
template <typename... Args>
std::string format(std::string fmt, Args... args);
```

with this inline definition:

```cpp
template <typename... Args>
inline std::string format(std::string fmt, Args... args) {
    size_t bufferSize = 1000;
    char* buffer = new char[bufferSize];
    int n = sprintf(buffer, fmt.c_str(), args...);
    assert(n >= 0 && n < (int)bufferSize - 1);
    std::string fmtStr(buffer);
    delete[] buffer;
    return fmtStr;
}
```

**Edit 2: `src/cpp/src/gguf_utils/gguf.cpp`, lines 16-27.** Delete the
duplicate definition (which is now in the header). The deleted block is:

```cpp
template <typename... Args>
std::string format(std::string fmt, Args... args) {
    size_t bufferSize = 1000;
    char* buffer = new char[bufferSize];
    int n = sprintf(buffer, fmt.c_str(), args...);
    assert(n >= 0 && n < (int)bufferSize - 1);

    std::string fmtStr(buffer);
    delete[] buffer;
    return fmtStr;
}

```

Leave the `#include "gguf_utils/gguf.hpp"` at the top of the file — it
now pulls the definition in.

**Verify the patch shape** before building:

```sh
cd /mnt/Ironwolf-4TB/openvino-hetero-test/openvino.genai
git diff --stat src/cpp/src/gguf_utils/
# expect: gguf.cpp | 12 ------------
#         gguf.hpp |  9 ++++++++-
```

### 9b. Build the wheel

Build. On a 16 GB box, **do not use `$(nproc)`** for this step — the genai
object library compiles about a dozen whisper + LLM transformation TUs that
each eat >1 GB at link time and the build OOMs. Cap parallelism at 2:

```sh
cd /mnt/Ironwolf-4TB/openvino-hetero-test/openvino.genai
OpenVINO_DIR=/mnt/Ironwolf-4TB/openvino-hetero-test/.venv/lib/python3.12/site-packages/openvino/cmake \
CMAKE_BUILD_PARALLEL_LEVEL=2 \
  pip wheel . --no-deps --no-build-isolation \
  --wheel-dir /tmp/genai_wheel
cd /mnt/Ironwolf-4TB/openvino-hetero-test

uv pip install --force-reinstall --no-deps \
  /tmp/genai_wheel/openvino_genai-*.whl
```

If you have >=32 GB RAM you can drop the `CMAKE_BUILD_PARALLEL_LEVEL=2`.

## 10. Verify

```sh
python3 -c "import openvino; print(openvino.__version__)"
python3 -c "import openvino_tokenizers; print(openvino_tokenizers.__version__)"
python3 -c "import openvino_genai; print(openvino_genai.__version__)"
```

## Smoke test (ov::Tensor boundary crossing)

```sh
python3 -c "
import numpy as np
import openvino as ov
import openvino_genai as ovg
t = ov.Tensor(np.zeros((1, 4), dtype=np.int64))
ti = ovg.TokenizedInputs(t, t)
assert ti.input_ids.shape == [1, 4]
print('ov.Tensor boundary crossing: OK')
"
```

## Sandbox verification status

Run against `/home/user/` (a 4-core / 15 GB sandbox, not `/mnt/Ironwolf-4TB/`):

| Step | Status |
|------|--------|
| 1 — clone | OK (both repos at master) |
| 2 — install_build_dependencies.sh | OK (only `nlohmann-json3-dev` missing) |
| 3 — uv venv --python 3.12 --seed | OK (pip 26.0.1 seeded) |
| 4 — requirements-dev + numpy + pybind11-stubgen | OK |
| 5 — cmake configure | OK (pybind11 v3.0.2, Python3 3.12.3) |
| 6 — cmake --build --target ie_wheel | OK (~50 min, wheel lands) |
| 7 — install OV wheel | OK (`openvino==2026.2.0`) |
| 8 — tokenizers wheel (py-build-cmake 0.4.3, cache wiped) | OK (`openvino-tokenizers==2026.2.0.0`) |
| 9a — apply gguf `format` template patch | Required on gcc 13; fixes dlopen-time undefined symbol. |
| 9b — genai wheel (py-build-cmake 0.5.0, `CMAKE_BUILD_PARALLEL_LEVEL=2`) | OK after patch (`openvino-genai==2026.2.0.0`). At default parallelism `$(nproc)=4` the build silently SIGKILLs at metadata-prep with no log output — OOM during concurrent cc1plus invocations. |
| 10 — version prints | OK |
| smoke test | OK |

## What to try if things break

### `undefined symbol: _Z6formatIJiE...__cxx11...basic_string...`

Not an ABI mismatch (even though the mangled name contains `__cxx11`). It's
the template-linkage bug covered in step 9a. Apply the patch, wipe
`.py-build-cmake_cache/`, rebuild. Quick verification from your built .so:

```sh
nm /path/to/openvino.genai/.py-build-cmake_cache/cp312-cp312-linux_x86_64/openvino_genai/libopenvino_genai.so.2620 \
  | c++filt | grep ' format<'
```

Before the patch you will see both a `U format<int>(...)` (undefined) and a
`t format<int>(...) [clone .isra.0]` (TU-local ISRA clone). After the patch,
`format<int>` becomes a `W` weak symbol — linker can resolve cross-TU refs.

### genai build OOMs even with `CMAKE_BUILD_PARALLEL_LEVEL=2`

1. Drop to `CMAKE_BUILD_PARALLEL_LEVEL=1`. Builds serially, ~3× slower but
   will fit in any box that can build OpenVINO itself.
2. Add swap: `sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`.
   The link phase spikes briefly — swap absorbs it without slowing
   steady-state compile.
3. Build at `-O2 -g0` instead of the default `-O3 -g` to shrink TUs:
   `CMAKE_ARGS="-DCMAKE_CXX_FLAGS=-O2 -DCMAKE_BUILD_TYPE=Release" pip wheel ...`.

### Real `_GLIBCXX_USE_CXX11_ABI` mismatch (if you ever get one)

Rare in this workflow because you're building OV and genai in the same
venv with the same compiler. If you see it, usually it's because some
other pip install pulled in the PyPI manylinux_2_28 OV wheel (ABI=0) on
top of your source-built wheel (ABI=1). Check with:

```sh
nm -D /path/to/.venv/lib/python3.12/site-packages/openvino/libs/libopenvino.so.2620 \
  | grep -c __cxx11
```

`0` → PyPI ABI=0 wheel snuck in. `~800` → source build, ABI=1 (what you want).
Fix: re-`pip install --force-reinstall --no-deps /path/to/your/source/wheel`.

## Notes

- `--seed` on `uv venv` is required because CMake's wheel build step and
  `pip wheel` both need pip in the venv.
- `--no-deps` skips dependency resolution — it does **not** skip metadata
  generation, so it does not bypass the py-build-cmake 0.5.0 PEP 639 check.
  (That was a misconception in the earlier draft.)
- `--no-build-isolation` prevents pip from creating a temporary venv, so the
  build links against the same OpenVINO installed in your venv. This prevents
  ABI mismatch.
- `OpenVINO_DIR` points at the cmake config installed by the OpenVINO wheel
  inside the venv. This is how tokenizers and GenAI find OpenVINO's headers
  and libraries.
- **Always wipe `.py-build-cmake_cache/` before rebuilding** if you changed
  `OpenVINO_DIR`, venv, or Python version — py-build-cmake does not detect
  these changes and will build against whatever was cached.
