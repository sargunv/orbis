//! A packed tile as a job worker transfers it to the map
//! (`orb_runtime::pack`): its header read, and its payload's spans checked
//! against the buffer, whatever the bytes.

#![no_main]
#![forbid(unsafe_code)]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    orb_fuzz::unpack(data);
});
