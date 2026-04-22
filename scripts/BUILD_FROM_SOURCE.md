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
| 9 — genai wheel (py-build-cmake 0.5.0, `CMAKE_BUILD_PARALLEL_LEVEL=2`) | **In progress** — last heartbeat at `[86%]`, 12 GB / 16 GB RSS, compiling `src/whisper/transformations/scaled_dot_product_attention_decomposition.cpp`. Earlier attempts at default parallelism `$(nproc)=4` silently SIGKILL'd at the metadata-prep stage with no log output — almost certainly OOM during concurrent cc1plus invocations. |
| 10 — version prints | not yet reached |
| smoke test | not yet reached |

## What to try next if genai OOMs again

1. Drop parallelism further: `CMAKE_BUILD_PARALLEL_LEVEL=1`. Builds serially,
   takes ~3× longer but will fit in any box that can build OpenVINO itself.
2. Add swap: `sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`.
   The genai link phase spikes briefly — swap absorbs it without slowing steady-state compile.
3. Build with `-g0 -O2` instead of the default `-O3 -g` to shrink TUs:
   `CMAKE_ARGS="-DCMAKE_CXX_FLAGS=-O2 -DCMAKE_BUILD_TYPE=Release" pip wheel ...`
4. If you hit the `ImportError: libopenvino_genai.so.2620: undefined symbol:
   _Z6format...__cxx11...basic_string...` after a successful build, that's the
   `_GLIBCXX_USE_CXX11_ABI` mismatch — your loaded libopenvino was built with
   ABI=0 and genai with ABI=1 (or vice versa). Fix: build both from source in
   the same venv (this doc), or rebuild genai with
   `-D_GLIBCXX_USE_CXX11_ABI=0` to match a PyPI manylinux_2_28- wheel.

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
