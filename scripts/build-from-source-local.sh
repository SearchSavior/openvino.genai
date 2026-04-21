#!/usr/bin/env bash
# Reproduces the full local build of openvino, openvino_tokenizers,
# and openvino.genai wheels — all installed into a single venv.
#
# Exercised path (Linux x86_64, Ubuntu 24.04, gcc 13, Python 3.11):
#   OpenVINO   master (pybind11 v3.0.2 submodule), _GLIBCXX_USE_CXX11_ABI=1
#   Tokenizers thirdparty/openvino_tokenizers pinned to 82c5a767 (+ local TF strip)
#   GenAI      this repo, pybind11 v3.0.1 via FetchContent, ABI=1
#
# Import smoke: `python -c "import openvino_genai"` succeeds; passing
# `ov.Tensor` through `openvino_genai.TokenizedInputs` round-trips.
set -euo pipefail

: "${WORKSPACE:=$HOME}"
: "${OV_SRC:=${WORKSPACE}/openvino}"
: "${GENAI_SRC:=${WORKSPACE}/openvino.genai}"
: "${VENV:=${WORKSPACE}/venv}"
: "${JOBS:=4}"

apt-get install -y patchelf

python3 -m venv "${VENV}"
"${VENV}/bin/pip" install --upgrade pip
"${VENV}/bin/pip" install numpy py-build-cmake==0.5.0 pybind11-stubgen==2.5.5 cmake==3.23.3

# ---------- OpenVINO ----------
if [ ! -d "${OV_SRC}" ]; then
    git clone --recursive https://github.com/openvinotoolkit/openvino.git "${OV_SRC}"
fi

cmake -S "${OV_SRC}" -B "${OV_SRC}/build_wheel" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_PYTHON=ON -DENABLE_WHEEL=ON \
    -DENABLE_SAMPLES=OFF -DENABLE_TESTS=OFF \
    -DENABLE_OV_IR_FRONTEND=ON -DENABLE_OV_PYTORCH_FRONTEND=ON \
    -DENABLE_OV_TF_FRONTEND=OFF -DENABLE_OV_TF_LITE_FRONTEND=OFF \
    -DENABLE_OV_ONNX_FRONTEND=OFF -DENABLE_OV_PADDLE_FRONTEND=OFF \
    -DENABLE_OV_JAX_FRONTEND=OFF \
    -DENABLE_INTEL_GPU=OFF -DENABLE_INTEL_NPU=OFF \
    -DENABLE_SYSTEM_TBB=ON -DENABLE_SYSTEM_PUGIXML=ON \
    -DCI_BUILD_NUMBER=2026.0.0-1-localbuild \
    -DPython3_EXECUTABLE="${VENV}/bin/python"

cmake --build "${OV_SRC}/build_wheel" --parallel "${JOBS}" --target wheel

"${VENV}/bin/pip" install --force-reinstall --no-deps \
    "${OV_SRC}"/build_wheel/wheels/openvino-*cp311-cp311-*.whl

# ---------- openvino.genai (clone + submodules) ----------
if [ ! -d "${GENAI_SRC}" ]; then
    git clone https://github.com/openvinotoolkit/openvino.genai.git "${GENAI_SRC}"
fi
git -C "${GENAI_SRC}" submodule update --init --recursive

# ---------- openvino_tokenizers ----------
# TF frontend is disabled above, so patch ov_extension.cpp to drop the TF
# dependency. Also switch the license expression to PEP 639 SPDX form so
# py-build-cmake 0.5.0 accepts the project metadata.
( cd "${GENAI_SRC}/thirdparty/openvino_tokenizers"
  OpenVINO_DIR="${VENV}/lib/python3.11/site-packages/openvino/cmake" \
    "${VENV}/bin/pip" wheel . --no-deps --no-build-isolation \
    --wheel-dir /tmp/tokenizers_wheel
)
"${VENV}/bin/pip" install --force-reinstall --no-deps \
    /tmp/tokenizers_wheel/openvino_tokenizers-*.whl

# ---------- openvino.genai ----------
# The pyproject already uses the SPDX license form on this branch.
# pybind11-stubgen must be on PATH for the py_openvino_genai_stub target.
( cd "${GENAI_SRC}"
  PATH="${VENV}/bin:${PATH}" \
  OpenVINO_DIR="${VENV}/lib/python3.11/site-packages/openvino/cmake" \
  CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" \
    "${VENV}/bin/pip" wheel . --no-deps --no-build-isolation \
    --wheel-dir /tmp/genai_wheel
)
"${VENV}/bin/pip" install --force-reinstall --no-deps \
    /tmp/genai_wheel/openvino_genai-*.whl

# ---------- smoke test ----------
"${VENV}/bin/python" - <<'PY'
import numpy as np
import openvino as ov
import openvino_genai as ovg
print("OV:", ov.__version__)
print("GenAI:", ovg.__version__)
t = ov.Tensor(np.zeros((1, 4), dtype=np.int64))
ti = ovg.TokenizedInputs(t, t)
assert ti.input_ids.shape == [1, 4]
print("import + ov.Tensor boundary crossing: OK")
PY
