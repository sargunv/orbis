# 0029: Device access through devices-cli

Android checks needed a device, and a machine that runs several worktrees side
by side has one emulator between them at best: suites that install an app or
open Chrome fight over it, and a script that names no serial gets whichever
device adb lists first. This records how the repository asks for one.

## What it does

[devices-cli](https://github.com/sargunv/devices-cli) gives each git worktree
its own test device, starts it on demand, caps how many run at once and stops it
when the worktree goes away. `mise.toml` pins it (built from a Go commit, so Go
is pinned beside it; it runs on Linux and macOS), and
`ORB_ANDROID_EMULATOR_API` names the one API level the repository's Android
checks use (plan D9's floor is API 24; the emulator runs 36).

`scripts/android-serial` is the single place that decides a serial, so the
shell, Node and Rust callers cannot drift apart:

- `ANDROID_SERIAL` wins when a caller set one;
- otherwise `devices-cli get android --api $ORB_ANDROID_EMULATOR_API
  --variant google_apis` boots or reuses this worktree's emulator and prints its
  serial.

`cargo xtask android --run`, `pnpm -C web bench --android` and
`pnpm -C web measure --android` all go through it, and each one passes `-s` to
every adb call, so no script picks "the" device. The bench and measure scripts
lost their `--android=<serial>` form: a serial in a command line is a
hard-coded device, and the environment already carries one.

The emulator is the Google APIs image rather than plain AOSP, because the bench
and measure scripts drive Chrome and the AOSP image has none. Its instance is
ephemeral, so Chrome starts there with its first-run screen skipped (a command
line file, which the image's debuggable build reads) and with
`--enable-unsafe-webgpu`, since its GPU is a software adapter; a phone keeps the
Chrome its owner set up.

What the emulator cannot do is render. Its software Vulkan draws only the
reference app's background, and Chrome's GPU process crashes inside
`vulkan.ranchu.so` before a WebGPU device exists (#54), so the emulator checks
the build, the install and the load path, and rendering fidelity and
performance stay on a phone.

## Physical phones

Functional checks do not need a phone, because the implementation is universal;
phones are for performance numbers from real slow hardware (a moto g's Adreno
619, a Galaxy Z Fold 7's Adreno 830), which is rarely worth taking. A
measurement takes `--physical`, which fails when the resolved device is an
emulator rather than falling back to one, and the caller serializes the phones
across agents by assigning one to its worktree for the measurement alone:

```sh
devices-cli assign <serial>
pnpm -C web bench --physical
devices-cli release
```

`AGENTS.md` carries the same rules: never hard-code a serial, never change
device settings (brightness, stay-awake) and never force-stop an app you did
not start.
