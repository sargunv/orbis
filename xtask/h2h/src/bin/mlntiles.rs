//! `mlntiles <style> --list tiles.txt [--size px] [--still]`: each
//! `[name] z x y` tile in `MapMode::Tile` (or with `--still`,
//! `MapMode::Static` at the tile's view), one map for the whole run, as
//! `orbis tiles --list --times` does. Prints
//! `tile <name> <z> <x> <y> <wall ms> <cpu ms>`, CPU being the process's.

use std::rc::Rc;
use std::time::Instant;

use mln_rs::mln_actor::RunLoop;
use mln_rs::mln_map::{Map, MapMode, MapOptions, NullMapObserver};
use mln_rs::mln_platform_default::HeadlessFrontend;
use mln_rs::mln_util::size::Size;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let style = args.first().expect("a style URL");
    let list = mln_h2h::flag(&args, "--list", "");
    let tiles: Vec<(String, u8, u32, u32)> = std::fs::read_to_string(&list)
        .expect("--list file")
        .lines()
        .map(|l| l.split('#').next().unwrap_or_default().trim())
        .filter(|l| !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let (name, n) = match f[..] {
                [z, x, y] => (format!("z{z}"), [z, x, y]),
                [name, z, x, y] => (name.to_string(), [z, x, y]),
                _ => panic!("{list}: want [name] z x y, got {l}"),
            };
            let num = |s: &str| s.parse::<u32>().expect("a tile coordinate");
            (name, num(n[0]) as u8, num(n[1]), num(n[2]))
        })
        .collect();
    let px: u32 = mln_h2h::flag(&args, "--size", "512")
        .parse()
        .expect("--size px");
    let still = args.iter().any(|a| a == "--still");
    let size = Size::new(px, px);
    let _run_loop = RunLoop::new();
    let mut frontend = HeadlessFrontend::new(size, 1.0);
    let mode = if still {
        MapMode::Static
    } else {
        MapMode::Tile
    };
    let mut map = Map::new(
        &frontend,
        Rc::new(NullMapObserver),
        MapOptions::new()
            .with_map_mode(mode)
            .with_size(size)
            .with_pixel_ratio(1.0),
    );
    map.get_style_mut().load_url(style);
    let started = Instant::now();
    for (name, z, x, y) in &tiles {
        map.jump_to(&mln_h2h::tile_camera(*z, *x, *y, px));
        let (wall, cpu) = (Instant::now(), mln_h2h::cpu());
        mln_h2h::render(&mut frontend, &mut map);
        let (wall, cpu) = (wall.elapsed(), mln_h2h::cpu() - cpu);
        println!(
            "tile {name} {z} {x} {y} {:.2} {:.2}",
            mln_h2h::ms(wall),
            mln_h2h::ms(cpu)
        );
    }
    eprintln!(
        "mlntiles: {} tiles in {:.1} s",
        tiles.len(),
        started.elapsed().as_secs_f64()
    );
}
