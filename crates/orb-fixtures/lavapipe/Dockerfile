# Toolchain matches rust-toolchain.toml. No window system, browser or GPU device.
FROM rust:1.99.1-slim-trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev libvulkan1 mesa-vulkan-drivers ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_TOOLCHAIN=1.99.1 RUSTUP_AUTO_INSTALL=0 WGPU_BACKEND=vulkan ORB_FIXTURES_LAVAPIPE=1 XDG_RUNTIME_DIR=/tmp
WORKDIR /work
COPY entrypoint.sh /usr/local/bin/orbis-lavapipe
ENTRYPOINT ["orbis-lavapipe"]
