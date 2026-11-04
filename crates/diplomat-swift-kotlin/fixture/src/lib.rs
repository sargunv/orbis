//! A bridge exercising every shape `diplomat-swift-kotlin` supports. The
//! golden tests read it; the run tests build it and call it from Swift and
//! Kotlin.

#[diplomat::bridge]
pub mod ffi {
    use std::fmt::Write;

    use diplomat_runtime::{DiplomatStr, DiplomatWrite};

    /// Why a call failed.
    #[diplomat::attr(auto, error)]
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug)]
    pub enum FixtureError {
        /// The argument was bad.
        Bad = 1,
        Worse = 2,
    }

    impl FixtureError {
        /// The calling thread's last error.
        #[diplomat::attr(contract, immediate)]
        #[diplomat::attr(contract, error_message)]
        pub fn last_message(out: &mut DiplomatWrite) {
            crate::LAST.with(|m| out.write_str(&m.borrow()).ok());
        }
    }

    #[derive(Debug)]
    pub enum Color {
        Red = 0,
        Green = 5,
    }

    #[derive(Debug)]
    pub struct Inner {
        pub a: u8,
        pub color: Color,
    }

    /// A value with every kind of field.
    #[derive(Debug)]
    pub struct Sample {
        pub x: f64,
        pub y: i32,
        pub big: u64,
        pub size: usize,
        pub flag: bool,
        pub inner: Inner,
    }

    /// A counter that belongs to one thread.
    #[diplomat::opaque_mut]
    #[diplomat::attr(contract, thread = "owner")]
    #[derive(Debug, Default)]
    pub struct Counter {
        pub(crate) count: i64,
        pub(crate) name: String,
        pub(crate) bytes: Vec<u8>,
        pub(crate) wake: Option<crate::Wake>,
    }

    impl Counter {
        #[diplomat::attr(contract, lifecycle)]
        pub fn create(start: i64) -> Box<Counter> {
            Box::new(Counter {
                count: start,
                bytes: vec![1, 2, 255],
                ..Default::default()
            })
        }

        /// Adds and returns the new count.
        #[diplomat::attr(contract, command)]
        pub fn add(&mut self, by: i64) -> i64 {
            self.count += by;
            self.count
        }

        #[diplomat::attr(contract, snapshot)]
        pub fn get(&self) -> i64 {
            self.count
        }

        #[diplomat::attr(contract, snapshot)]
        pub fn name(&self, out: &mut DiplomatWrite) {
            out.write_str(&self.name).ok();
        }

        /// Fails on an empty name.
        #[diplomat::attr(contract, command)]
        pub fn set_name(&mut self, name: &DiplomatStr) -> Result<(), FixtureError> {
            if name.is_empty() {
                return Err(crate::fail(FixtureError::Bad, "the name is empty"));
            }
            self.name = String::from_utf8_lossy(name).into_owned();
            Ok(())
        }

        /// Writes the name, or fails without one.
        #[diplomat::attr(contract, query)]
        pub fn checked_name(&self, out: &mut DiplomatWrite) -> Result<(), FixtureError> {
            if self.name.is_empty() {
                return Err(crate::fail(FixtureError::Worse, "no name"));
            }
            out.write_str(&self.name).ok();
            Ok(())
        }

        #[diplomat::attr(contract, query)]
        pub fn sum(&self, values: &[f64]) -> f64 {
            values.iter().sum()
        }

        #[diplomat::attr(contract, snapshot)]
        #[allow(clippy::needless_lifetimes)] // Diplomat needs them named.
        pub fn bytes<'a>(&'a self) -> &'a [u8] {
            &self.bytes
        }

        #[diplomat::attr(contract, immediate)]
        pub fn echo(&self, point: Sample) -> Sample {
            point
        }

        #[diplomat::attr(contract, query)]
        pub fn divide(&self, by: i64) -> Result<i64, FixtureError> {
            if by == 0 {
                return Err(crate::fail(FixtureError::Bad, "division by zero"));
            }
            Ok(self.count / by)
        }

        #[diplomat::attr(contract, query)]
        #[allow(clippy::result_unit_err)] // A unit error is a shape under test.
        pub fn checked(&self, ok: bool) -> Result<u32, ()> {
            if ok { Ok(7) } else { Err(()) }
        }

        #[diplomat::attr(contract, lifecycle)]
        pub fn item(&self) -> Box<Item> {
            Box::new(Item(self.count))
        }

        #[diplomat::attr(contract, query)]
        pub fn find(&self, present: bool) -> Option<Box<Item>> {
            present.then(|| Box::new(Item(42)))
        }

        #[diplomat::attr(contract, command)]
        pub fn absorb(&mut self, item: &Item) -> i64 {
            self.count += item.0;
            self.count
        }

        /// Keeps `wake` to call from another thread in `fire`.
        #[diplomat::attr(contract, command)]
        pub fn set_wake(&mut self, wake: impl Fn() + 'static) {
            self.wake = Some(crate::Wake(Box::new(wake)));
        }

        /// Calls the wake callback from a new thread, `times` times.
        #[diplomat::attr(contract, command)]
        pub fn fire(&self, times: u32) {
            if let Some(wake) = &self.wake {
                std::thread::scope(|s| {
                    s.spawn(|| {
                        for _ in 0..times {
                            (wake.0)();
                        }
                    });
                });
            }
        }

        #[diplomat::attr(contract, snapshot)]
        pub fn color(&self) -> Color {
            if self.count % 2 == 0 {
                Color::Red
            } else {
                Color::Green
            }
        }
    }

    /// A value any thread may use.
    #[diplomat::opaque]
    #[diplomat::attr(contract, thread = "any")]
    #[derive(Debug)]
    pub struct Item(pub(crate) i64);

    impl Item {
        #[diplomat::attr(contract, snapshot)]
        pub fn value(&self) -> i64 {
            self.0
        }

        /// The value after `ms` milliseconds: a call long enough to race.
        #[diplomat::attr(contract, query)]
        pub fn slow_value(&self, ms: u32) -> i64 {
            std::thread::sleep(std::time::Duration::from_millis(u64::from(ms)));
            self.0
        }
    }
}

use std::cell::RefCell;

thread_local! {
    static LAST: RefCell<String> = const { RefCell::new(String::new()) };
}

fn fail(failure: ffi::FixtureError, message: &str) -> ffi::FixtureError {
    LAST.with(|m| *m.borrow_mut() = message.to_owned());
    failure
}

/// A callback the bindings promise can run on any thread.
pub(crate) struct Wake(Box<dyn Fn()>);

impl std::fmt::Debug for Wake {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Wake")
    }
}
// SAFETY: the bridge's contract for `set_wake` is that the callback may be
// called from any thread; the bindings pass thread-safe closures.
unsafe impl Sync for Wake {}
// SAFETY: as above.
unsafe impl Send for Wake {}
