//! Separate serial diagnostic pass. Absent from primary timing builds.
//! Backend spans are nested inside shared master transport spans; consumers
//! must subtract them from transport before forming an additive breakdown.
use super::{guard, write};
use ninjars::MasterIntegral;
use std::{cell::RefCell, time::Instant};

pub const RECORDS: usize = 49;
const CELLS: usize = RECORDS * 3;
struct State {
    enabled: bool,
    values: [u64; CELLS],
}
thread_local! {
    static STATE: RefCell<State> = const { RefCell::new(State { enabled: false, values: [0; CELLS] }) };
}

pub(crate) struct Scope(Option<(Instant, usize, usize)>);
impl Scope {
    fn new(kind: usize, rows: usize) -> Self {
        Self(STATE.with(|state| state.borrow().enabled.then(|| (Instant::now(), kind, rows))))
    }
    pub(crate) fn numerator(kind: u32, rows: usize) -> Self {
        if kind < 4 {
            Self::new(kind as usize, rows)
        } else {
            Self(None)
        }
    }
    pub(crate) fn preparation(rows: usize) -> Self {
        Self::new(4, rows)
    }
    pub(crate) fn master(integral: MasterIntegral, source: u8, rows: usize) -> Self {
        let kind = match integral {
            MasterIntegral::A0 => 0,
            MasterIntegral::B0 => 1,
            MasterIntegral::B1 => 2,
            MasterIntegral::B11 => 3,
            MasterIntegral::B111 => 4,
            MasterIntegral::C0 => 5,
            MasterIntegral::D0 => 6,
            MasterIntegral::DB0 => 7,
            MasterIntegral::An { rank } => 8 + rank.min(5),
        };
        assert!(
            (1..=3).contains(&source),
            "invalid master diagnostic source"
        );
        Self::new(5 + 14 * usize::from(source - 1) + kind, rows)
    }
    pub(crate) fn master_wire(source: u8, rows: usize) -> Self {
        if source == 1 || source == 2 {
            Self::new(47 + usize::from(source - 1), rows)
        } else {
            Self(None)
        }
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        if let Some((start, kind, rows)) = self.0 {
            let nanos = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            STATE.with(|state| {
                let mut state = state.borrow_mut();
                if state.enabled {
                    state.values[kind * 3] = state.values[kind * 3].saturating_add(1);
                    state.values[kind * 3 + 1] =
                        state.values[kind * 3 + 1].saturating_add(rows as u64);
                    state.values[kind * 3 + 2] = state.values[kind * 3 + 2].saturating_add(nanos);
                }
            });
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sp_execution_begin() -> i32 {
    guard(|| {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            if state.enabled {
                return Err("execution diagnostics already active".into());
            }
            state.values.fill(0);
            state.enabled = true;
            Ok(())
        })
    })
}

/// # Safety
/// `RECORDS*3` writable cells; serial calls on the same thread as begin.
/// Columns are calls, rows and inclusive elapsed nanoseconds. Records0..3 are
/// numerator direct/Mu/T3/T2;4 preparation;5..46 are14 master kinds for shared
/// Rust, shared original and direct Rust;47/48 shared Rust/original transport.
/// Master kinds are A0,B0,B1,B11,B111,C0,D0,DB0,An0..4 and An_other.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_execution_end(out: *mut u64, len: usize) -> i32 {
    guard(|| {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            if !state.enabled {
                return Err("execution diagnostics are not active".into());
            }
            unsafe { write(out, len, &state.values) }?;
            state.enabled = false;
            Ok(())
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn higher_tensor_rank_is_counted_and_invalid_source_is_rejected() {
        assert_eq!(sp_execution_begin(), 0);
        {
            let _scope = Scope::master(MasterIntegral::An { rank: 17 }, 3, 2);
        }
        let mut values = [0; CELLS];
        assert_eq!(
            unsafe { sp_execution_end(values.as_mut_ptr(), values.len()) },
            0
        );
        assert_eq!(&values[46 * 3..46 * 3 + 2], &[1, 2]);
        assert!(std::panic::catch_unwind(|| Scope::master(MasterIntegral::A0, 0, 1)).is_err());
    }

    #[test]
    fn scopes_distinguish_backend_from_transport_and_count_rows() {
        assert_eq!(sp_execution_begin(), 0);
        assert_ne!(sp_execution_begin(), 0);
        {
            let _wire = Scope::master_wire(1, 8);
            let _backend = Scope::master(MasterIntegral::D0, 1, 8);
        }
        {
            let _lookup = Scope::master_wire(2, 1);
        }
        {
            let _num = Scope::numerator(2, 32);
        }
        let mut values = [0; CELLS];
        assert_eq!(
            unsafe { sp_execution_end(values.as_mut_ptr(), values.len()) },
            0
        );
        assert_eq!(&values[47 * 3..47 * 3 + 2], &[1, 8]);
        assert_eq!(&values[48 * 3..48 * 3 + 2], &[1, 1]);
        assert_eq!(&values[11 * 3..11 * 3 + 2], &[1, 8]);
        assert!(values[47 * 3 + 2] >= values[11 * 3 + 2]);
        assert!(values[19 * 3..33 * 3].iter().all(|v| *v == 0));
        assert_eq!(&values[6..8], &[1, 32]);
        assert_ne!(
            unsafe { sp_execution_end(values.as_mut_ptr(), values.len()) },
            0
        );
    }
}
