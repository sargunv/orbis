//! orb-hal descriptors → wgpu descriptors.

use orb_hal::*;

pub(crate) fn buffer_usages(usage: BufferUsages) -> wgpu::BufferUsages {
    let mut out = wgpu::BufferUsages::empty();
    for (from, to) in [
        (BufferUsages::COPY_SRC, wgpu::BufferUsages::COPY_SRC),
        (BufferUsages::COPY_DST, wgpu::BufferUsages::COPY_DST),
        (BufferUsages::INDEX, wgpu::BufferUsages::INDEX),
        (BufferUsages::UNIFORM, wgpu::BufferUsages::UNIFORM),
        (BufferUsages::STORAGE, wgpu::BufferUsages::STORAGE),
        (BufferUsages::INDIRECT, wgpu::BufferUsages::INDIRECT),
    ] {
        if usage.contains(from) {
            out |= to;
        }
    }
    out
}

pub(crate) fn texture_usages(usage: TextureUsages) -> wgpu::TextureUsages {
    let mut out = wgpu::TextureUsages::empty();
    for (from, to) in [
        (TextureUsages::COPY_SRC, wgpu::TextureUsages::COPY_SRC),
        (
            TextureUsages::TRANSIENT_ATTACHMENT,
            wgpu::TextureUsages::TRANSIENT_ATTACHMENT,
        ),
        (TextureUsages::COPY_DST, wgpu::TextureUsages::COPY_DST),
        (
            TextureUsages::TEXTURE_BINDING,
            wgpu::TextureUsages::TEXTURE_BINDING,
        ),
        (
            TextureUsages::STORAGE_BINDING,
            wgpu::TextureUsages::STORAGE_BINDING,
        ),
        (
            TextureUsages::RENDER_ATTACHMENT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        ),
    ] {
        if usage.contains(from) {
            out |= to;
        }
    }
    out
}

pub(crate) fn shader_stages(stages: ShaderStages) -> wgpu::ShaderStages {
    let mut out = wgpu::ShaderStages::empty();
    for (from, to) in [
        (ShaderStages::VERTEX, wgpu::ShaderStages::VERTEX),
        (ShaderStages::FRAGMENT, wgpu::ShaderStages::FRAGMENT),
        (ShaderStages::COMPUTE, wgpu::ShaderStages::COMPUTE),
    ] {
        if stages.contains(from) {
            out |= to;
        }
    }
    out
}

pub(crate) fn texture_format(format: TextureFormat) -> wgpu::TextureFormat {
    match format {
        TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
        TextureFormat::R32Float => wgpu::TextureFormat::R32Float,
        TextureFormat::R32Uint => wgpu::TextureFormat::R32Uint,
        TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
    }
}

/// The orb-hal format for a wgpu format, if orb-hal has one.
pub fn from_texture_format(format: wgpu::TextureFormat) -> Option<TextureFormat> {
    Some(match format {
        wgpu::TextureFormat::Rgba16Float => TextureFormat::Rgba16Float,
        wgpu::TextureFormat::R8Unorm => TextureFormat::R8Unorm,
        wgpu::TextureFormat::R32Float => TextureFormat::R32Float,
        wgpu::TextureFormat::R32Uint => TextureFormat::R32Uint,
        wgpu::TextureFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba8UnormSrgb => TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Bgra8Unorm => TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Bgra8UnormSrgb => TextureFormat::Bgra8UnormSrgb,
        wgpu::TextureFormat::Depth32Float => TextureFormat::Depth32Float,
        _ => return None,
    })
}

pub(crate) fn bind_group_layout_entry(entry: &BindGroupLayoutEntry) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: entry.binding,
        visibility: shader_stages(entry.visibility),
        ty: match entry.ty {
            BindingType::Uniform => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            BindingType::Storage { read_only } => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            BindingType::Texture {
                filterable,
                dimension,
            } => wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: view_dimension(dimension),
                multisampled: false,
            },
            BindingType::Sampler { filtering } => wgpu::BindingType::Sampler(if filtering {
                wgpu::SamplerBindingType::Filtering
            } else {
                wgpu::SamplerBindingType::NonFiltering
            }),
            BindingType::DataTexture => wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            BindingType::StorageTexture { format, dimension } => {
                wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: texture_format(format),
                    view_dimension: view_dimension(dimension),
                }
            }
        },
        count: None,
    }
}

pub(crate) fn color_target(target: &ColorTargetState) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format: texture_format(target.format),
        blend: target.blend.map(|blend| match blend {
            BlendState::Additive => wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            BlendState::PremultipliedAlpha => wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            BlendState::Max => {
                let max = wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Max,
                };
                wgpu::BlendState {
                    color: max,
                    alpha: max,
                }
            }
            BlendState::Multiply => wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::Src,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
        }),
        write_mask: wgpu::ColorWrites::ALL,
    }
}

pub(crate) fn view_dimension(dimension: TextureViewDimension) -> wgpu::TextureViewDimension {
    match dimension {
        TextureViewDimension::D2 => wgpu::TextureViewDimension::D2,
        TextureViewDimension::D2Array => wgpu::TextureViewDimension::D2Array,
    }
}

pub(crate) fn sampler(desc: &SamplerDesc<'_>) -> wgpu::SamplerDescriptor<'static> {
    let filter = match desc.filter {
        FilterMode::Nearest => wgpu::FilterMode::Nearest,
        FilterMode::Linear => wgpu::FilterMode::Linear,
    };
    wgpu::SamplerDescriptor {
        anisotropy_clamp: desc.anisotropy.max(1),
        mag_filter: filter,
        min_filter: filter,
        mipmap_filter: match desc.mipmap_filter {
            FilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
            FilterMode::Linear => wgpu::MipmapFilterMode::Linear,
        },
        ..Default::default()
    }
}

/// Bytes per texel of a format that can be written.
pub(crate) fn texel_bytes(format: wgpu::TextureFormat) -> u32 {
    format
        .block_copy_size(None)
        .expect("writable textures have a block size")
}
