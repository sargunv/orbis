# Toolchain matches rust-toolchain.toml. No window system, browser or GPU
# device: Mesa's lavapipe (Vulkan, Modern) and llvmpipe (GLES 3 through
# surfaceless EGL, Compat). `docker run` picks one with
# ORB_FIXTURES_LAVAPIPE or ORB_FIXTURES_LLVMPIPE.
FROM rust:1.99.1-slim-trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev libvulkan1 mesa-vulkan-drivers libegl1 libegl-mesa0 libgles2 \
    libgl1-mesa-dri ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_TOOLCHAIN=1.99.1 RUSTUP_AUTO_INSTALL=0 WGPU_BACKEND=vulkan XDG_RUNTIME_DIR=/tmp \
    EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1
WORKDIR /work
COPY entrypoint.sh /usr/local/bin/orbis-lavapipe
ENTRYPOINT ["orbis-lavapipe"]
