//! Opt-in Rust allocation diagnostic. Excluded from primary timing builds.
//! C++ and native-library allocations that bypass Rust GlobalAlloc are not seen.
use super::{guard, write};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
static ENABLED: AtomicBool = AtomicBool::new(false);
static TOTAL: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static SCOPES: [[AtomicU64; 5]; 3] = [const { [const { AtomicU64::new(0) }; 5] }; 3];
struct Diagnostic;
#[global_allocator]
static ALLOCATOR: Diagnostic = Diagnostic;
fn count(index: usize, value: u64) {
    if ENABLED.load(Ordering::Relaxed) {
        TOTAL[index].fetch_add(value, Ordering::Relaxed);
    }
}
// SAFETY: all operations delegate directly to System with the unchanged
// pointer/layout contract. Counters are atomic and never allocate recursively.
unsafe impl GlobalAlloc for Diagnostic {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(0, 1);
        count(1, layout.size() as u64);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(0, 1);
        count(1, layout.size() as u64);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count(2, 1);
        count(3, size as u64);
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        count(4, 1);
        unsafe { System.dealloc(pointer, layout) }
    }
}
pub(crate) fn snapshot() -> [u64; 5] {
    std::array::from_fn(|i| TOTAL[i].load(Ordering::Relaxed))
}
pub(crate) fn record(scope: usize, before: [u64; 5]) {
    if ENABLED.load(Ordering::Relaxed) {
        for (i, value) in snapshot().into_iter().enumerate() {
            SCOPES[scope][i].fetch_add(value - before[i], Ordering::Relaxed);
        }
    }
}
pub(crate) struct Scope {
    category: usize,
    before: [u64; 5],
}
impl Scope {
    pub(crate) fn new(category: usize) -> Self {
        Self {
            category,
            before: snapshot(),
        }
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        record(self.category, self.before);
    }
}
/// Start exclusive serial diagnostics; not safe to overlap independent scopes.
#[unsafe(no_mangle)]
pub extern "C" fn sp_allocation_begin() -> i32 {
    guard(|| {
        if ENABLED.load(Ordering::Relaxed) {
            return Err("allocation diagnostic already active".into());
        }
        for counter in TOTAL.iter().chain(SCOPES.iter().flatten()) {
            counter.store(0, Ordering::Relaxed);
        }
        ENABLED.store(true, Ordering::Relaxed);
        Ok(())
    })
}
/// # Safety
/// Twenty valid writable cells. Entries are totals, numerator, master, physical
/// prep; each has alloc calls/bytes, realloc calls/bytes, dealloc calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_allocation_end(out: *mut u64, len: usize) -> i32 {
    guard(|| {
        if !ENABLED.swap(false, Ordering::Relaxed) {
            return Err("allocation diagnostic is not active".into());
        }
        let mut values = [0; 20];
        values[..5].copy_from_slice(&snapshot());
        for (scope, counters) in SCOPES.iter().enumerate() {
            for (i, counter) in counters.iter().enumerate() {
                values[5 + scope * 5 + i] = counter.load(Ordering::Relaxed);
            }
        }
        unsafe { write(out, len, &values) }
    })
}
