//! Validation-only spans injected into a copied source tree by instrument.py.
//! No algebra, no production feature, no inference from these timings to parity.
use std::{cell::RefCell, collections::BTreeMap, fmt::Write, panic::Location, time::Instant};

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
struct Key { kind: &'static str, label: &'static str, parent: &'static str, file: &'static str, line: u32, inputs: usize, outputs: usize }
#[derive(Default)]
struct Counts { calls: u64, rows: u64, inclusive_ns: u128, exclusive_ns: u128 }
struct Frame { label: &'static str, children_ns: u128 }
#[derive(Default)]
struct State { enabled: bool, frames: Vec<Frame>, counts: BTreeMap<Key, Counts> }
thread_local! { static STATE: RefCell<State> = RefCell::new(State::default()); }
pub struct Span { timing: Option<(Instant, Key, usize)> }
impl Span {
    #[track_caller]
    pub fn kernel(rows: usize, inputs: usize, outputs: usize) -> Self {
        Self::new("kernel", "exact-evaluator", rows, inputs, outputs, Location::caller())
    }
    pub fn phase(label: &'static str, rows: usize) -> Self {
        Self::new("phase", label, rows, 0, 0, Location::caller())
    }
    fn new(kind: &'static str, label: &'static str, rows: usize, inputs: usize, outputs: usize, location: &'static Location<'static>) -> Self {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            if !state.enabled { return Self { timing: None }; }
            let parent = state.frames.last().map_or("none", |frame| frame.label);
            state.frames.push(Frame { label, children_ns: 0 });
            Self { timing: Some((Instant::now(), Key { kind, label, parent, file: location.file(), line: location.line(), inputs, outputs }, rows)) }
        })
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        if let Some((start, key, rows)) = self.timing {
            let elapsed = start.elapsed().as_nanos();
            STATE.with(|state| {
                let mut state = state.borrow_mut();
                let frame = state.frames.pop().expect("diagnostic span stack");
                assert_eq!(frame.label, key.label);
                if let Some(parent) = state.frames.last_mut() { parent.children_ns += elapsed; }
                let count = state.counts.entry(key).or_default();
                count.calls += 1; count.rows += rows as u64;
                count.inclusive_ns += elapsed;
                count.exclusive_ns += elapsed.saturating_sub(frame.children_ns);
            });
        }
    }
}
pub fn start() {
    STATE.with(|state| { let mut state = state.borrow_mut(); assert!(state.frames.is_empty()); state.counts.clear(); state.enabled = true; });
}
pub fn stop() -> String {
    STATE.with(|state| {
        let mut state = state.borrow_mut(); assert!(state.frames.is_empty()); state.enabled = false;
        let mut text = String::from("{\"performance_acceptance\":false,\"records\":[");
        for (i, (key, count)) in state.counts.iter().enumerate() {
            if i > 0 { text.push(','); }
            write!(&mut text,"{{\"kind\":{:?},\"label\":{:?},\"parent\":{:?},\"file\":{:?},\"line\":{},\"inputs\":{},\"outputs\":{},\"calls\":{},\"rows\":{},\"inclusive_ns\":{},\"exclusive_ns\":{}}}",key.kind,key.label,key.parent,key.file,key.line,key.inputs,key.outputs,count.calls,count.rows,count.inclusive_ns,count.exclusive_ns).unwrap();
        }
        text.push_str("]}\n"); text
    })
}
