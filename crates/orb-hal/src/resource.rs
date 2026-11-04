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
    /// A set of timestamp queries.
    QuerySetId;
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
        const VERTEX = 1 << 3;
        const UNIFORM = 1 << 4;
        const STORAGE = 1 << 5;
        const INDIRECT = 1 << 6;
        const QUERY_RESOLVE = 1 << 7;
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct TextureUsages: u32 {
        const COPY_SRC = 1 << 0;
        const COPY_DST = 1 << 1;
        const TEXTURE_BINDING = 1 << 2;
        const STORAGE_BINDING = 1 << 3;
        const RENDER_ATTACHMENT = 1 << 4;
        /// Attachment contents stay within one render pass (memoryless on TBDR).
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
pub enum IndexFormat {
    Uint16,
    Uint32,
}

/// Timestamp queries, for devices with [`Caps::timestamp_query`].
///
/// [`Caps::timestamp_query`]: crate::Caps::timestamp_query
#[derive(Clone, Debug, PartialEq)]
pub struct QuerySetDesc<'a> {
    pub label: &'a str,
    pub count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    R8Unorm,
    /// Filterable, blendable half-float density targets.
    Rgba16Float,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    Depth32Float,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextureDesc<'a> {
    pub label: &'a str,
    pub width: u32,
    pub height: u32,
    pub array_layers: u32,
    pub format: TextureFormat,
    pub usage: TextureUsages,
    pub sample_count: u32,
}

/// A rectangle of one layer of a texture, for writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureRegion {
    pub texture: TextureId,
    pub x: u32,
    pub y: u32,
    pub layer: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterMode {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
}

/// A sampler without mipmaps: one filter for magnification and
/// minification, one address mode for both axes.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplerDesc<'a> {
    pub label: &'a str,
    pub filter: FilterMode,
    pub address: AddressMode,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShaderModuleDesc<'a> {
    pub label: &'a str,
    pub wgsl: Cow<'a, str>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingType {
    Uniform,
    Storage {
        read_only: bool,
    },
    /// A sampled texture.
    Texture {
        sample: TextureSampleType,
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
pub enum TextureSampleType {
    Float { filterable: bool },
    Uint,
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
    pub fn buffer(buffer: BufferId) -> Self {
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
    /// Front faces use counterclockwise winding.
    pub cull_mode: Option<Face>,
    pub sample_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Face {
    Front,
    Back,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthState {
    pub format: TextureFormat,
    pub write: bool,
    pub compare: CompareFunction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareFunction {
    Always,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
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
