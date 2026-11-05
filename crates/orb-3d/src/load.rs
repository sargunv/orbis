//! Loading a model from its URL, without I/O: [`Load`] says what to fetch
//! (external buffers, then images by URI, resolved against the document),
//! takes the bytes back, hands out the encoded images for the host to
//! decode, and packs the model for the GPU.

use std::collections::HashMap;

use crate::gltf::Gltf;
use crate::gpu::{GpuModel, pack};
use crate::{ImageSource, Model};

/// One model loading.
#[derive(Debug)]
pub struct Load {
    url: String,
    gltf: Gltf,
    /// Buffer index by resolved URL, and what arrived.
    buffers: Vec<(usize, String)>,
    fetched: HashMap<String, Vec<u8>>,
    model: Option<Model>,
}

impl Load {
    /// The document fetched from `url`: a `.gltf` or `.glb`.
    pub fn new(url: &str, bytes: &[u8]) -> Result<Self, String> {
        let gltf = Gltf::parse(bytes)?;
        let buffers = gltf
            .external_buffers()
            .into_iter()
            .map(|(i, uri)| (i, join(url, &uri)))
            .collect();
        Ok(Self {
            url: url.to_string(),
            gltf,
            buffers,
            fetched: HashMap::new(),
            model: None,
        })
    }

    /// URLs to fetch before the next step: the external buffers, then,
    /// once the model is built from them, its images given by URI.
    pub fn wanted(&self) -> Vec<String> {
        let urls: Vec<String> = match &self.model {
            None => self.buffers.iter().map(|(_, url)| url.clone()).collect(),
            Some(model) => model
                .images
                .iter()
                .filter_map(|image| match image {
                    ImageSource::Uri(uri) => Some(join(&self.url, uri)),
                    ImageSource::Bytes(_) => None,
                })
                .collect(),
        };
        let mut urls: Vec<String> = urls
            .into_iter()
            .filter(|u| !self.fetched.contains_key(u))
            .collect();
        urls.sort();
        urls.dedup();
        urls
    }

    /// A wanted URL's bytes.
    pub fn fetched(&mut self, url: &str, bytes: Vec<u8>) {
        self.fetched.insert(url.to_string(), bytes);
    }

    /// Once nothing is [`wanted`](Self::wanted) for the buffers, builds
    /// the model; once nothing is wanted for its images either, returns
    /// them encoded, by image index, for the host to decode (`None` for an
    /// image that failed to fetch). `Ok(None)` while fetches are pending.
    pub fn images(&mut self) -> Result<Option<Vec<Option<Vec<u8>>>>, String> {
        if self.model.is_none() {
            if !self.wanted().is_empty() {
                return Ok(None);
            }
            let external = self
                .buffers
                .iter()
                .filter_map(|(i, url)| Some((*i, self.fetched.remove(url)?)))
                .collect();
            self.model = Some(self.gltf.build(&external)?);
        }
        if !self.wanted().is_empty() {
            return Ok(None);
        }
        let model = self.model.as_ref().expect("built above");
        Ok(Some(
            model
                .images
                .iter()
                .map(|image| match image {
                    ImageSource::Bytes(bytes) => Some(bytes.clone()),
                    ImageSource::Uri(uri) => self.fetched.get(&join(&self.url, uri)).cloned(),
                })
                .collect(),
        ))
    }

    /// The model packed for the GPU, given its images decoded (width,
    /// height, straight RGBA8) by index; images that failed leave their
    /// material slots untextured.
    pub fn finish(self, decoded: &[Option<(u32, u32, Vec<u8>)>]) -> Result<GpuModel, String> {
        let model = self.model.ok_or("the model isn't built yet")?;
        Ok(pack(&model, decoded))
    }
}

/// A glTF URI against the document's URL (RFC 3986's reference
/// resolution, for the paths glTF uses).
pub fn join(base: &str, uri: &str) -> String {
    if uri.contains("://") || uri.starts_with("data:") {
        return uri.to_string();
    }
    let (origin, path) = match base.find("://") {
        Some(scheme) => match base[scheme + 3..].find('/') {
            Some(slash) => base.split_at(scheme + 3 + slash),
            None => (base, ""),
        },
        None => ("", base),
    };
    if uri.starts_with('/') {
        return format!("{origin}{uri}");
    }
    let path = path.split(['?', '#']).next().unwrap_or_default();
    let mut parts: Vec<&str> = path.split('/').collect();
    parts.pop();
    for segment in uri.split('/') {
        match segment {
            "." => {}
            ".." => {
                if parts.len() > 1 {
                    parts.pop();
                }
            }
            s => parts.push(s),
        }
    }
    format!("{origin}{}", parts.join("/"))
}
