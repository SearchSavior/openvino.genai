# CI-matching build containers

Two Dockerfiles that mirror the two containers the openvino.genai CI uses
for wheel builds, plus a `docker-compose.yml` to drive them.

| Service          | Base                                   | Compiler       | Matches CI |
|------------------|----------------------------------------|----------------|------------|
| `ubuntu-22-04`   | `ubuntu:22.04` + deadsnakes            | gcc 11         | `linux.yml` nightly genai wheel build |
| `manylinux-2-28` | `quay.io/pypa/manylinux_2_28_x86_64`   | gcc 14, `-fno-lto` | `manylinux_2_28.yml` PyPI wheel build |

Use these when you need to reproduce CI behavior — e.g. debugging a wheel
that fails on your host but works from PyPI, or confirming the gguf
`format` template patch is actually required on your gcc (it's required
on gcc 13, not on either CI profile).

## Build and enter

From this directory (`scripts/docker/`):

```sh
# Build once. Ubuntu image is ~2 min, manylinux is ~30 sec (base is heavier).
docker compose build ubuntu-22-04         # or: manylinux-2-28

# Interactive shell. Repo root is mounted at /workspace/openvino.genai.
docker compose run --rm ubuntu-22-04
```

Source is mounted read/write — any `.py-build-cmake_cache/`, wheel
outputs etc. land on the host. Wipe caches when switching between host
and container builds:

```sh
rm -rf /workspace/openvino.genai/.py-build-cmake_cache
rm -rf /workspace/openvino.genai/thirdparty/openvino_tokenizers/.py-build-cmake_cache
```

## Build the wheels inside

The CI-faithful flow uses **PEP 517 build isolation** (no
`--no-build-isolation`), so each sub-wheel pulls its pinned
`py-build-cmake` version automatically. No need for the manual
0.4.3 ↔ 0.5.0 dance that the non-isolated flow in `BUILD_FROM_SOURCE.md`
requires.

OpenVINO itself is built in a separate CI job and downloaded as an
artifact. For local dev, two options:

1. **Use a PyPI OV wheel** (fastest, matches what PyPI users get).
   Inside the container:
   ```sh
   /opt/python/cp312-cp312/bin/python -m venv /tmp/venv        # manylinux
   # OR
   python3.12 -m venv /tmp/venv                                # ubuntu
   source /tmp/venv/bin/activate
   pip install --pre -U openvino
   ```

2. **Build OV from source inside the same container** — follow steps 5-7
   of `../BUILD_FROM_SOURCE.md` verbatim. Only the manylinux container
   produces a PyPI-grade wheel; the Ubuntu container produces an
   Ubuntu-22.04-glibc wheel that is fine for local use.

With OV installed, build tokenizers + genai:

```sh
cd /workspace/openvino.genai/thirdparty/openvino_tokenizers
pip wheel . --no-deps --wheel-dir /tmp/tokenizers_wheel \
    --config-settings=cmake.args="-DOpenVINO_DIR=$(python -c 'import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), "cmake"))')"
pip install --force-reinstall --no-deps /tmp/tokenizers_wheel/openvino_tokenizers-*.whl

cd /workspace/openvino.genai
pip wheel . --no-deps --wheel-dir /tmp/genai_wheel \
    --config-settings=cmake.args="-DOpenVINO_DIR=$(python -c 'import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), "cmake"))')"
pip install --force-reinstall --no-deps /tmp/genai_wheel/openvino_genai-*.whl
```

## Reproducing the exact CI `pip wheel` invocation

The CI command from `.github/workflows/linux.yml:360-368`:

```sh
pip wheel -v --no-deps --wheel-dir /tmp/wheels \
    --config-settings=override=cross.arch="manylinux_2_31_x86_64" \
    --config-settings=cmake.args="-DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_C_COMPILER_LAUNCHER=sccache" \
    openvino                                       `# CI's pre-built OV wheel URL` \
    /workspace/openvino.genai
```

The `override=cross.arch="manylinux_2_31_x86_64"` tag writes the wheel's
platform tag to match what CI publishes. Drop it if you only want the
wheel to run locally — the native tag (e.g. `manylinux_2_39_x86_64` on
Ubuntu 24, `manylinux_2_28_x86_64` on the manylinux container) is more
accurate.

## Is the gguf `format` patch still required inside these containers?

No. Neither `gcc 11` (ubuntu-22-04) nor `gcc 14 + -fno-lto` (manylinux-2-28)
ISRA-clones the `format<int>` template hard enough to drop its external
symbol. The patch in step 9a of `../BUILD_FROM_SOURCE.md` is only needed
on gcc 13 (the Ubuntu 24.04 / Fedora-40-era default).

You can confirm with `nm` on any build you produce here:

```sh
nm /workspace/openvino.genai/.py-build-cmake_cache/cp312-cp312-linux_x86_64/openvino_genai/libopenvino_genai.so.2620 \
  | c++filt | grep ' format<int>'
```

Should show a `W` (weak) or `T` (external) entry with no unresolved `U`.
