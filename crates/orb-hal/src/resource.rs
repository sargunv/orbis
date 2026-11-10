use std::borrow::Cow;

macro_rules! ids {
    ($($(#[$doc:meta])* $name:ident;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u32);
    )*};
}

ids! {
    BufferId;
    /// A texture, or a target the host registered, such as a surface frame.
    TextureId;
    ShaderModuleId;
    BindGroupLayoutId;
    BindGroupId;
    RenderPipelineId;
    ComputePipelineId;
    RenderBundleId;
    SamplerId;
    /// Bytes the host holds outside core memory, such as a transferred JS
    /// `ArrayBuffer`, registered with the implementation.
    HostBytesId;
    /// A decoded image the host holds outside core memory, such as a
    /// transferred JS `ImageBitmap`, registered with the implementation.
    HostImageId;
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct BufferUsages: u32 {
        const COPY_SRC = 1 << 0;
        const COPY_DST = 1 << 1;
        const INDEX = 1 << 2;
        const UNIFORM = 1 << 3;
        const STORAGE = 1 << 4;
        const INDIRECT = 1 << 5;
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct TextureUsages: u32 {
        const COPY_SRC = 1 << 0;
        const COPY_DST = 1 << 1;
        const TEXTURE_BINDING = 1 << 2;
        const STORAGE_BINDING = 1 << 3;
        const RENDER_ATTACHMENT = 1 << 4;
        /// Attachment contents stay within one render pass (memoryless on
        /// TBDR). A hint: dropped where the platform lacks it.
        const TRANSIENT_ATTACHMENT = 1 << 5;
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct ShaderStages: u32 {
        const VERTEX = 1 << 0;
        const FRAGMENT = 1 << 1;
        const COMPUTE = 1 << 2;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BufferDesc<'a> {
    pub label: &'a str,
    pub size: u64,
    pub usage: BufferUsages,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    R8Unorm,
    /// Unfilterable single floats: the terrain's depth as a colour.
    R32Float,
    /// Filterable, blendable half-float density targets.
    Rgba16Float,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    Depth32Float,
}

impl TextureFormat {
    /// Bytes a texel takes.
    pub fn texel_bytes(self) -> u32 {
        match self {
            Self::R8Unorm => 1,
            Self::Rgba16Float => 8,
            Self::R32Float
            | Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb
            | Self::Depth32Float => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextureDesc<'a> {
    pub label: &'a str,
    pub width: u32,
    pub height: u32,
    pub array_layers: u32,
    /// How shaders sample it: every binding uses this dimension.
    pub dimension: TextureViewDimension,
    pub format: TextureFormat,
    pub usage: TextureUsages,
    pub sample_count: u32,
    /// Mip levels, at least 1. Level 0 is the full size.
    pub mip_levels: u32,
    /// Formats its views may take besides `format`: its sRGB twin, to
    /// draw and sample in linear light while storage writes the encoding
    /// ([`Gpu::create_texture_view_as`](crate::Gpu::create_texture_view_as)).
    pub view_formats: &'a [TextureFormat],
}

impl TextureFormat {
    /// The format with sRGB encoding on writes and decoding on reads, for
    /// 8-bit colour; others are their own.
    pub fn srgb(self) -> Self {
        match self {
            Self::Rgba8Unorm => Self::Rgba8UnormSrgb,
            Self::Bgra8Unorm => Self::Bgra8UnormSrgb,
            other => other,
        }
    }

    pub fn is_srgb(self) -> bool {
        matches!(self, Self::Rgba8UnormSrgb | Self::Bgra8UnormSrgb)
    }
}

/// A rectangle of one layer of a texture's mip level, for writes; its
/// position and size are in that level's texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureRegion {
    pub texture: TextureId,
    pub x: u32,
    pub y: u32,
    pub layer: u32,
    pub width: u32,
    pub height: u32,
    pub level: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterMode {
    Nearest,
    Linear,
}

/// One filter for magnification and minification, and how mip levels
/// blend; coordinates clamp to the edge.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplerDesc<'a> {
    pub label: &'a str,
    pub filter: FilterMode,
    /// Between mip levels: `Nearest` samples one level (all a texture
    /// without mipmaps has), `Linear` blends the two nearest.
    pub mipmap_filter: FilterMode,
    /// The most samples along a footprint's long axis: 1 is isotropic.
    /// Above 1 every filter must be `Linear`.
    pub anisotropy: u16,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShaderModuleDesc<'a> {
    pub label: &'a str,
    /// WGSL.
    pub source: Cow<'a, str>,
    /// `source` as naga modules, one per entry point and postcard-encoded,
    /// when the build made them (`orb_shaders::Shader::ir`): a device that
    /// compiles with naga makes each stage's module from these rather than
    /// parse `source`.
    pub ir: Option<&'static [(&'static str, &'static [u8])]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BindGroupLayoutDesc<'a> {
    pub label: &'a str,
    pub entries: &'a [BindGroupLayoutEntry],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindGroupLayoutEntry {
    pub binding: u32,
    pub visibility: ShaderStages,
    pub ty: BindingType,
}

impl BindGroupLayoutEntry {
    pub const fn uniform(binding: u32, visibility: ShaderStages) -> Self {
        Self {
            binding,
            visibility,
            ty: BindingType::Uniform,
        }
    }

    /// A storage buffer shaders only read.
    pub const fn storage(binding: u32, visibility: ShaderStages) -> Self {
        Self {
            binding,
            visibility,
            ty: BindingType::Storage { read_only: true },
        }
    }

    /// A storage buffer shaders write.
    pub const fn storage_rw(binding: u32, visibility: ShaderStages) -> Self {
        Self {
            binding,
            visibility,
            ty: BindingType::Storage { read_only: false },
        }
    }

    pub const fn texture(
        binding: u32,
        visibility: ShaderStages,
        filterable: bool,
        dimension: TextureViewDimension,
    ) -> Self {
        Self {
            binding,
            visibility,
            ty: BindingType::Texture {
                filterable,
                dimension,
            },
        }
    }

    pub const fn sampler(binding: u32, visibility: ShaderStages, filtering: bool) -> Self {
        Self {
            binding,
            visibility,
            ty: BindingType::Sampler { filtering },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingType {
    Uniform,
    Storage {
        read_only: bool,
    },
    /// A sampled float texture.
    Texture {
        filterable: bool,
        dimension: TextureViewDimension,
    },
    Sampler {
        filtering: bool,
    },
    /// A write-only storage texture.
    StorageTexture {
        format: TextureFormat,
        dimension: TextureViewDimension,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureViewDimension {
    D2,
    D2Array,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BindGroupDesc<'a> {
    pub label: &'a str,
    pub layout: BindGroupLayoutId,
    pub entries: &'a [BindGroupEntry],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindGroupEntry {
    pub binding: u32,
    pub resource: BindingResource,
}

impl BindGroupEntry {
    /// The whole of `buffer`.
    pub const fn buffer(binding: u32, buffer: BufferId) -> Self {
        Self {
            binding,
            resource: BindingResource::buffer(buffer),
        }
    }

    /// `size` bytes of `buffer` from `offset`.
    pub const fn slice(binding: u32, buffer: BufferId, offset: u64, size: u64) -> Self {
        Self {
            binding,
            resource: BindingResource::Buffer {
                buffer,
                offset,
                size: Some(size),
            },
        }
    }

    pub const fn texture(
        binding: u32,
        texture: TextureId,
        dimension: TextureViewDimension,
    ) -> Self {
        Self {
            binding,
            resource: BindingResource::Texture { texture, dimension },
        }
    }

    pub const fn sampler(binding: u32, sampler: SamplerId) -> Self {
        Self {
            binding,
            resource: BindingResource::Sampler(sampler),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingResource {
    /// A buffer range; `size: None` binds to the end.
    Buffer {
        buffer: BufferId,
        offset: u64,
        size: Option<u64>,
    },
    /// A view of every layer of a texture, sampled or as storage.
    Texture {
        texture: TextureId,
        dimension: TextureViewDimension,
    },
    Sampler(SamplerId),
}

impl BindingResource {
    /// The whole buffer.
    pub const fn buffer(buffer: BufferId) -> Self {
        Self::Buffer {
            buffer,
            offset: 0,
            size: None,
        }
    }
}

/// A shader entry point with values for its `override` constants.
#[derive(Clone, Debug, PartialEq)]
pub struct ShaderStage<'a> {
    pub module: ShaderModuleId,
    pub entry_point: &'a str,
    pub constants: &'a [(&'a str, f64)],
}

#[derive(Clone, Debug, PartialEq)]
pub struct RenderPipelineDesc<'a> {
    pub label: &'a str,
    pub bind_group_layouts: &'a [BindGroupLayoutId],
    /// Vertex input is pulled from storage buffers; there are no vertex
    /// buffers.
    pub vertex: ShaderStage<'a>,
    pub fragment: ShaderStage<'a>,
    pub targets: &'a [ColorTargetState],
    pub depth: Option<DepthState>,
    pub topology: PrimitiveTopology,
    /// Whether back faces are culled; front faces wind counterclockwise.
    pub cull_back: bool,
    pub sample_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveTopology {
    LineList,
    TriangleList,
    TriangleStrip,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorTargetState {
    pub format: TextureFormat,
    pub blend: Option<BlendState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendState {
    /// Source colours are premultiplied by alpha.
    PremultipliedAlpha,
    /// ONE + ONE, for Gaussian density accumulation.
    Additive,
}

/// A depth test in reversed Z: a fragment passes at or nearer than the
/// depth held (`greater-equal`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthState {
    pub format: TextureFormat,
    pub write: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComputePipelineDesc<'a> {
    pub label: &'a str,
    pub bind_group_layouts: &'a [BindGroupLayoutId],
    pub compute: ShaderStage<'a>,
}

/// The attachment formats a bundle is replayed against.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderBundleDesc<'a> {
    pub label: &'a str,
    pub color_formats: &'a [TextureFormat],
    pub depth_format: Option<TextureFormat>,
    pub sample_count: u32,
}
