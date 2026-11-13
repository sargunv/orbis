//! Giving the GPU back: what a renderer holds goes when it does.
//! Each piece releases its own resources; ids it only binds (the shared
//! image atlas, the textures it was handed) stay.

use rustc_hash::FxHashMap as HashMap;
use std::collections::BTreeMap;

use orb_hal::*;

pub(crate) trait Release {
    fn release(self, gpu: &mut dyn Gpu);
}

macro_rules! ids {
    ($($id:ty => |$gpu:ident, $v:ident| $e:expr;)*) => {
        $(impl Release for $id {
            fn release(self, $gpu: &mut dyn Gpu) {
                let $v = self;
                $e
            }
        })*
    };
}

ids! {
    BufferId => |gpu, id| gpu.destroy_buffer(id);
    TextureId => |gpu, id| gpu.destroy_texture(id);
    BindGroupId => |gpu, id| gpu.destroy_bind_group(id);
    BindGroupLayoutId => |gpu, id| gpu.destroy_bind_group_layout(id);
    SamplerId => |gpu, id| gpu.destroy_sampler(id);
    RenderBundleId => |gpu, id| gpu.destroy_render_bundle(id);
    RenderPipelineId => |gpu, id| gpu.destroy_pipeline(id.into());
    ComputePipelineId => |gpu, id| gpu.destroy_pipeline(id.into());
}

impl<T: Release> Release for Option<T> {
    fn release(self, gpu: &mut dyn Gpu) {
        if let Some(v) = self {
            v.release(gpu);
        }
    }
}

impl<T: Release> Release for Vec<T> {
    fn release(self, gpu: &mut dyn Gpu) {
        self.into_iter().for_each(|v| v.release(gpu));
    }
}

impl<T: Release, const N: usize> Release for [T; N] {
    fn release(self, gpu: &mut dyn Gpu) {
        self.into_iter().for_each(|v| v.release(gpu));
    }
}

impl<K, V: Release> Release for HashMap<K, V> {
    fn release(self, gpu: &mut dyn Gpu) {
        self.into_values().for_each(|v| v.release(gpu));
    }
}

impl<K, V: Release> Release for BTreeMap<K, V> {
    fn release(self, gpu: &mut dyn Gpu) {
        self.into_values().for_each(|v| v.release(gpu));
    }
}

/// Plain values a tuple of resources carries.
macro_rules! plain {
    ($($t:ty),*) => {
        $(impl Release for $t {
            fn release(self, _: &mut dyn Gpu) {}
        })*
    };
}

plain!(u32, usize);

macro_rules! tuples {
    ($(($($t:ident),+))*) => {
        $(impl<$($t: Release),+> Release for ($($t,)+) {
            #[allow(non_snake_case)]
            fn release(self, gpu: &mut dyn Gpu) {
                let ($($t,)+) = self;
                $($t.release(gpu);)+
            }
        })*
    };
}

tuples! {
    (A, B)
    (A, B, C)
    (A, B, C, D)
    (A, B, C, D, E)
    (A, B, C, D, E, F)
    (A, B, C, D, E, F, G)
    (A, B, C, D, E, F, G, H)
    (A, B, C, D, E, F, G, H, I)
    (A, B, C, D, E, F, G, H, I, J)
    (A, B, C, D, E, F, G, H, I, J, K)
    (A, B, C, D, E, F, G, H, I, J, K, L)
}
