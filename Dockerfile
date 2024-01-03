# ---------------------------------------------------------------
# ---------------------- BUILD MMDET/MMCV ----------------------
# ---------------------------------------------------------------
# Based on mmdet official Dockerfile
ARG PYTORCH_VERSION
ARG CUDA_VERSION
ARG CUDNN_VERSION
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel as built-mmdet

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1" \
		DEBIAN_FRONTEND=noninteractive

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMEngine and MMCV
RUN pip install openmim && \
    mim install "mmengine>=0.7.1" "mmcv>=2.0.0rc4"

# ------------------- (END) BUILD MMDET/MMCV --------------------

# ---------------------------------------------------------------
# --------------------- TRAIN ENVIRONNEMENT ---------------------
# ---------------------------------------------------------------
# Use of more lightweight image for development
# as CUDA compilation should not be necessary
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime AS train

ENV SITE_PACKAGES="/opt/conda/lib/python3.10/site-packages"
COPY --from=built-mmdet ${SITE_PACKAGES}/mmcv ${SITE_PACKAGES}/mmcv
COPY --from=built-mmdet ${SITE_PACKAGES}/mmcv-2.0.1.dist-info ${SITE_PACKAGES}/mmcv-2.0.1.dist-info
COPY --from=built-mmdet ${SITE_PACKAGES}/mmengine ${SITE_PACKAGES}/mmengine
COPY --from=built-mmdet ${SITE_PACKAGES}/mmengine-0.8.4.dist-info ${SITE_PACKAGES}/mmengine-0.8.4.dist-info
COPY --from=built-mmdet ${SITE_PACKAGES}/mim ${SITE_PACKAGES}/mim
COPY --from=built-mmdet ${SITE_PACKAGES}/openmim-0.3.9.dist-info ${SITE_PACKAGES}/openmim-0.3.9.dist-info

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y build-essential ffmpeg git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_ARGS="--no-cache-dir"
RUN pip install ${PIP_ARGS} --upgrade pip setuptools


COPY . .
ARG COMMIT
RUN echo "Training with commit: ${COMMIT}" && \
    git checkout --force ${COMMIT} && \
    pip install ${PIP_ARGS} .
# ------------------- (END) DEV ENVIRONNEMENT --------------------
