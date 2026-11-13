//! core.wasm's data by feature: a read-only object of the linked module (a
//! string, a table, a shader's source) that only one feature module's code
//! reaches, directly or through other such objects, goes with that
//! module, which writes it into its thread's memory as it's instantiated:
//! each thread's instance of core.wasm has a memory of its own, and no
//! code but the module's reads the object. core.wasm's copy is zeroed, and
//! binaryen drops the zeros.
//!
//! The linker's map bounds each object ([`objects`]), so a pointer
//! anywhere into one is a reference to all of it. A reference is any
//! constant of a function's code that points into an object, a load's or
//! store's offset included, and any aligned word of another object that
//! does: numbers that only look like addresses keep more in core.wasm,
//! never less. Mutable objects stay in core.wasm, and so does what they
//! point at.

use std::collections::HashSet;
use std::ops::Range;

/// A data object of the linked module, from the linker's map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Object {
    pub(crate) at: u32,
    pub(crate) size: u32,
    pub(crate) read_only: bool,
}

/// The data objects of a wasm-ld map (`--Map`), by address: each input
/// chunk of its data segments. Those of `.bss` hold no bytes.
pub(crate) fn objects(map: &str) -> Result<Vec<Object>, String> {
    let mut objects = Vec::new();
    for line in map.lines() {
        // `Addr Off Size Out In Symbol`: an input chunk names its file and
        // section, `path:(section)`.
        let mut fields = line.split_whitespace();
        let (Some(at), Some(_), Some(size), Some(input)) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let Some(section) = input
            .rsplit_once(":(")
            .and_then(|(_, s)| s.strip_suffix(')'))
        else {
            continue;
        };
        let read_only = section == ".rodata" || section.starts_with(".rodata.");
        let data = section == ".data" || section.starts_with(".data.");
        if !(read_only || data) {
            continue;
        }
        let hex = |s: &str| u32::from_str_radix(s, 16).map_err(|e| format!("map: {line}: {e}"));
        let (at, size) = (hex(at)?, hex(size)?);
        if size > 0 {
            objects.push(Object {
                at,
                size,
                read_only,
            });
        }
    }
    objects.sort_by_key(|o| o.at);
    if objects.windows(2).any(|w| w[0].at + w[0].size > w[1].at) {
        return Err("the linker's map has overlapping data objects".into());
    }
    Ok(objects)
}

/// The bytes of a module's active data segments: where each starts in
/// memory, and its bytes' range in the file.
pub(crate) type Segments = [(u32, Range<usize>)];

/// Where each object goes: `Some(m)` for feature module `m`, `None` for
/// core.wasm. `functions` is each function's pointers (its constants that
/// may be addresses) and module; `roots`, pointers of core.wasm's outside
/// any function (globals).
pub(crate) fn assign(
    objects: &[Object],
    wasm: &[u8],
    segments: &Segments,
    functions: &[(&[u32], Option<usize>)],
    roots: &[u32],
) -> Vec<Option<usize>> {
    let find = |address: u32| {
        let i = objects
            .partition_point(|o| o.at <= address)
            .checked_sub(1)?;
        let o = &objects[i];
        (address < o.at + o.size).then_some(i)
    };
    // What each object's aligned words point at.
    let points: Vec<Vec<usize>> = objects
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let mut to: Vec<usize> = (o.at.next_multiple_of(4)..o.at + o.size)
                .step_by(4)
                .filter(|a| a + 4 <= o.at + o.size)
                .filter_map(|a| word(wasm, segments, a))
                .filter_map(find)
                .filter(|&j| j != i)
                .collect();
            to.sort_unstable();
            to.dedup();
            to
        })
        .collect();
    let mut core = vec![false; objects.len()];
    let reach = |from: &mut Vec<usize>, seen: &mut Vec<bool>, stop: &[bool]| {
        while let Some(i) = from.pop() {
            for &j in &points[i] {
                if !seen[j] && !stop[j] {
                    seen[j] = true;
                    from.push(j);
                }
            }
        }
    };
    let mut start: Vec<usize> = (functions.iter())
        .filter(|(_, m)| m.is_none())
        .flat_map(|(p, _)| p.iter())
        .chain(roots)
        .filter_map(|&p| find(p))
        .chain((0..objects.len()).filter(|&i| !objects[i].read_only))
        .collect();
    for &i in &start {
        core[i] = true;
    }
    let none = vec![false; objects.len()];
    reach(&mut start, &mut core, &none);
    let modules = (functions.iter())
        .filter_map(|(_, m)| *m)
        .max()
        .map_or(0, |m| m + 1);
    loop {
        // What each module's code reaches past core.wasm's.
        let mut owners: Vec<HashSet<usize>> = vec![HashSet::new(); objects.len()];
        for m in 0..modules {
            let mut seen = vec![false; objects.len()];
            let mut from: Vec<usize> = (functions.iter())
                .filter(|(_, owner)| *owner == Some(m))
                .flat_map(|(p, _)| p.iter())
                .filter_map(|&p| find(p))
                .filter(|&i| !core[i])
                .collect();
            for &i in &from {
                seen[i] = true;
            }
            reach(&mut from, &mut seen, &core);
            for (i, seen) in seen.into_iter().enumerate() {
                if seen {
                    owners[i].insert(m);
                }
            }
        }
        // Two modules' objects stay in core.wasm, with what they reach.
        let mut shared: Vec<usize> = (0..objects.len())
            .filter(|&i| owners[i].len() > 1)
            .collect();
        if shared.is_empty() {
            return owners.into_iter().map(|o| o.into_iter().next()).collect();
        }
        for &i in &shared {
            core[i] = true;
        }
        reach(&mut shared, &mut core, &none);
    }
}

/// The little-endian word at `address` of the module's memory, if a
/// segment writes it.
fn word(wasm: &[u8], segments: &Segments, address: u32) -> Option<u32> {
    let (start, range) = segments
        .iter()
        .find(|(start, range)| *start <= address && address + 4 <= start + range.len() as u32)?;
    let at = range.start + (address - start) as usize;
    Some(u32::from_le_bytes(wasm[at..at + 4].try_into().ok()?))
}

/// `wasm` with `objects` zeroed in its data segments.
pub(crate) fn zero(wasm: &mut [u8], segments: &Segments, objects: &[Object]) {
    for o in objects {
        for (start, range) in segments {
            let (from, to) = (
                o.at.max(*start),
                (o.at + o.size).min(start + range.len() as u32),
            );
            if from < to {
                let at = range.start + (from - start) as usize;
                wasm[at..at + (to - from) as usize].fill(0);
            }
        }
    }
}

/// The bytes `objects` hold in `wasm`'s memory, as segments of
/// consecutive objects: where each starts, and its bytes.
pub(crate) fn bytes(wasm: &[u8], segments: &Segments, objects: &[Object]) -> Vec<(u32, Vec<u8>)> {
    let mut out: Vec<(u32, Vec<u8>)> = Vec::new();
    for o in objects {
        let mut held = vec![0; o.size as usize];
        for (start, range) in segments {
            let (from, to) = (
                o.at.max(*start),
                (o.at + o.size).min(start + range.len() as u32),
            );
            if from < to {
                let at = range.start + (from - start) as usize;
                held[(from - o.at) as usize..(to - o.at) as usize]
                    .copy_from_slice(&wasm[at..at + (to - from) as usize]);
            }
        }
        match out.last_mut() {
            Some((start, run)) if *start + run.len() as u32 == o.at => run.extend(held),
            _ => out.push((o.at, held)),
        }
    }
    out
}

/// `module` with a data section that writes `segments` into its memory
/// (its first, which a feature module imports from core.wasm), after its
/// code.
pub(crate) fn with_data(module: &[u8], segments: &[(u32, Vec<u8>)]) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    uleb(&mut body, segments.len() as u64);
    for (at, bytes) in segments {
        // Active, memory 0, at `i32.const at`.
        body.push(0);
        body.push(0x41);
        sleb(&mut body, i64::from(*at));
        body.push(0x0b);
        uleb(&mut body, bytes.len() as u64);
        body.extend_from_slice(bytes);
    }
    // After the code section (10), before what follows it.
    let mut at = 8;
    let mut insert = None;
    while at < module.len() {
        let id = module[at];
        let (size, len) = read_uleb(&module[at + 1..]).ok_or("a malformed section")?;
        let end = at + 1 + len + size as usize;
        if id == 11 {
            return Err("the module already has a data section".into());
        }
        if id == 10 {
            insert = Some(end);
        }
        at = end;
    }
    let insert = insert.unwrap_or(module.len());
    let mut out = module[..insert].to_vec();
    out.push(11);
    uleb(&mut out, body.len() as u64);
    out.extend_from_slice(&body);
    out.extend_from_slice(&module[insert..]);
    Ok(out)
}

fn uleb(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn sleb(out: &mut Vec<u8>, mut value: i64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        let done = (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0);
        if done {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// An unsigned LEB128 at the start of `bytes`, and its length.
fn read_uleb(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    for (i, &b) in bytes.iter().enumerate().take(10) {
        value |= u64::from(b & 0x7f) << (7 * i);
        if b & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_map_bounds_the_data_objects() {
        let map = "    Addr      Off     Size Out     In      Symbol
       -    3cbeed    55658 DATA
  100000   3cbeee    55170 .rodata
  100000   3cbef8      375e         <internal>:(.rodata)
  103760   3cf658       10         /x/a.o:(.rodata..Lanon.b9.1)
  103760   3cf658       10                 .Lanon.b9.1
  155170   421068      4d0 .data
  155170   421068        8         /x/a.o:(.data.X)
  155640   3cbeed      3e1 .bss
  155640        0       18         /x/a.o:(.bss.Y)";
        assert_eq!(
            objects(map).unwrap(),
            [
                Object {
                    at: 0x100000,
                    size: 0x375e,
                    read_only: true
                },
                Object {
                    at: 0x103760,
                    size: 0x10,
                    read_only: true
                },
                Object {
                    at: 0x155170,
                    size: 8,
                    read_only: false
                },
            ]
        );
    }

    #[test]
    fn an_object_goes_with_the_one_module_that_reaches_it() {
        let object = |at, size, read_only| Object {
            at,
            size,
            read_only,
        };
        // 0: a table that points at 1; 2: core's; 3: read by modules 0 and
        // 1, and points at 4; 5: mutable, module 1's; 6: what 5 points at.
        let objects = [
            object(0x100000, 8, true),
            object(0x100008, 4, true),
            object(0x10000c, 4, true),
            object(0x100010, 4, true),
            object(0x100014, 4, true),
            object(0x100018, 4, false),
            object(0x10001c, 4, true),
        ];
        let mut memory = Vec::new();
        for word in [0x100008u32, 0, 7, 9, 0x100014, 9, 0x10001c, 3] {
            memory.extend(word.to_le_bytes());
        }
        let segments = [(0x100000, 0..memory.len())];
        // Module 0 reads the table, at an offset into it, and 3; module 1
        // reads 3 and 5; core.wasm reads 2.
        let m0 = [0x100004, 0x100010];
        let m1 = [0x100010, 0x100018];
        let core = [0x10000c, 42];
        let functions: [(&[u32], Option<usize>); 3] =
            [(&m0, Some(0)), (&m1, Some(1)), (&core, None)];
        let homes = assign(&objects, &memory, &segments, &functions, &[]);
        assert_eq!(homes, [Some(0), Some(0), None, None, None, None, None]);
        let mut zeroed = memory.clone();
        let moved: Vec<Object> = (0..objects.len())
            .filter(|&i| homes[i] == Some(0))
            .map(|i| objects[i])
            .collect();
        zero(&mut zeroed, &segments, &moved);
        assert!(zeroed[..12].iter().all(|&b| b == 0));
        assert_eq!(zeroed[12..], memory[12..]);
        assert_eq!(
            bytes(&memory, &segments, &moved),
            [(0x100000, memory[..12].to_vec())]
        );
    }

    #[test]
    fn a_module_takes_its_data_after_its_code() {
        // Type, function and code sections of a module with one empty
        // function, then a custom section.
        let module = [
            0, b'a', b's', b'm', 1, 0, 0, 0, 1, 4, 1, 0x60, 0, 0, 3, 2, 1, 0, 10, 4, 1, 2, 0, 0x0b,
            0, 2, 1, b'x',
        ];
        let out = with_data(&module, &[(0x100000, vec![7, 8])]).unwrap();
        let data = [11, 11, 1, 0, 0x41, 0x80, 0x80, 0xc0, 0, 0x0b, 2, 7, 8];
        assert_eq!(out[..24], module[..24]);
        assert_eq!(out[24..24 + data.len()], data);
        assert_eq!(out[24 + data.len()..], module[24..]);
    }
}
