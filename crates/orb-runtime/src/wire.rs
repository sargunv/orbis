//! The one codec for messages between workers (decision 0044): postcard
//! over serde, for style updates, sprite icons, tile requests and replies,
//! placement messages and glyph bitmaps. Mesh and table streams travel
//! beside a message, raw, so they can be transferred.

use serde::Serialize;
use serde::de::DeserializeOwned;

/// A message's bytes.
pub fn encode<T: Serialize>(message: &T) -> Vec<u8> {
    postcard::to_allocvec(message).expect("messages encode")
}

/// A message from its bytes.
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    postcard::from_bytes(bytes).map_err(|e| format!("malformed message: {e}"))
}
