use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU8, Ordering};
use std::ptr::null_mut;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

struct ChunkIndices<const LEN: usize> {
    indices: AtomicU32,
}

impl<const LEN: usize> ChunkIndices<LEN> {
    pub const fn new() -> Self {
        Self {
            indices: AtomicU32::new(0),
        }
    }

    pub fn push(&self) -> Option<usize> {
        self
            .indices
            .fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |x| {
                    let end = x >> 16;
                    if end == LEN as u32 {
                        return None;
                    }
                    Some((end + 1) << 16 | (x & 0xffff))
                }
            )
            .ok()
            .map(|x| (x >> 16) as _)
    }

    pub fn pop(&self) -> Result<Option<usize>, ()> {
        match self
            .indices
            .fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |x| {
                    let start = (x & 0xffff) + 1;
                    let end = x >> 16;

                    if start > end || start > LEN as _ {
                        return None;
                    }

                    Some((end << 16) | start)
                }
            ) {
            Ok(x) => Ok(Some(x as usize & 0xffff)),
            Err(x) => {
                if x & 0xffff == LEN as u32 {
                    Err(())
                } else {
                    Ok(None)
                }
            }
        }
    }

    pub fn is_finished(&self) -> bool {
        let state = self
            .indices
            .load(Ordering::Relaxed);

        let start = state & 0xffff;

        start == LEN as u32
    }

    pub fn is_full(&self) -> bool {
        self
            .indices
            .load(Ordering::Relaxed)
            >> 16
            == LEN as u32
    }

    pub fn reset(&self) {
        self.indices.store(0, Ordering::Relaxed);
    }
}

const STATE_UNINIT: u8 = 0b000;
const STATE_MID_INIT: u8 = 0b001;
const STATE_INIT: u8 = 0b010;
const STATE_MID_UNINIT: u8 = 0b011;
const STATE_FINISHED: u8 = 0b100;

// SAFETY: The align of this struct may not change.
pub struct Chunk<T, const LEN: usize> {
    reset_state: AtomicU8,
    indices: ChunkIndices<LEN>,
    pub(crate) next: AtomicPtr<Chunk<T, LEN>>,
    #[cfg(debug_assertions)]
    pub(crate) prev: AtomicPtr<Chunk<T, LEN>>,
    chunk_data: [(UnsafeCell<MaybeUninit<T>>, AtomicU8); LEN],
}

unsafe impl<T: Send, const LEN: usize> Send for Chunk<T, LEN> {}

unsafe impl<T: Send, const LEN: usize> Sync for Chunk<T, LEN> {}

impl<T, const LEN: usize> Default for Chunk<T, LEN> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const LEN: usize> Chunk<T, LEN> {
    pub fn new() -> Self {
        assert!(LEN <= u16::MAX as usize);
        Self {
            indices: ChunkIndices::new(),
            reset_state: AtomicU8::new(STATE_INIT),
            next: AtomicPtr::new(null_mut()),
            chunk_data: [(); LEN].map(|_| {
                (
                    UnsafeCell::new(MaybeUninit::uninit()),
                    AtomicU8::new(STATE_UNINIT),
                )
            }),
            #[cfg(debug_assertions)]
            prev: AtomicPtr::new(null_mut()),
        }
    }

    /// * `Ok(Some(x))` -> Success.
    /// * `Ok(None)` -> No items.
    /// * `Err(())` -> Finished.
    pub fn pop(&self) -> Result<Option<T>, ()> {
        // let pop_idx = self.indices.pop_idx();
        let pop_idx = match self.indices.pop() {
            Ok(Some(x)) => x,
            Ok(None) => return Ok(None),
            Err(()) => return Err(()),
        };

        let (cell, state) = &self.chunk_data[pop_idx];

        while state.compare_exchange(
                STATE_INIT,
                STATE_MID_UNINIT,
                Ordering::Acquire,
                Ordering::Relaxed
            )
            .is_err() {}

        let val = unsafe {
            std::ptr::read((*cell.get()).as_ptr())
        };
        #[cfg(debug_assertions)]
        state
            .compare_exchange(
                STATE_MID_UNINIT,
                STATE_FINISHED,
                Ordering::Release,
                Ordering::Relaxed,
            )
            .unwrap();
        #[cfg(not(debug_assertions))]
        state.store(STATE_FINISHED, Ordering::Release);

        return Ok(Some(val));
    }

    pub fn push(&self, val: T) -> Result<(), T> {
        let push_idx = if let Some(x) = self.indices.push() {
            x
        } else {
            return Err(val);
        };

        let (cell, state) = &self.chunk_data[push_idx];
        unsafe { std::ptr::write(cell.get(), MaybeUninit::new(val)) };
        state.store(STATE_INIT, Ordering::Release);

        return Ok(())
    }

    pub fn is_finished(&self) -> bool {
        self.indices.is_finished()
    }

    pub fn is_full(&self) -> bool {
        self.indices.is_full()
    }

    pub fn try_reset(&self) -> Result<(), ()> {

        let res = if self.is_finished() {
            self
                .reset_state
                .compare_exchange(STATE_INIT, STATE_MID_UNINIT, Ordering::Acquire, Ordering::Relaxed)
                .map_err(|_| ())?;

            loop {
                if let Err(()) = self.chunk_data
                    .iter()
                    .map(|(_, flag)| {
                        loop {
                            match flag
                                .compare_exchange(STATE_FINISHED, STATE_UNINIT, Ordering::Relaxed, Ordering::Relaxed) {
                                Ok(_) => break Ok(()),
                                Err(STATE_MID_INIT | STATE_INIT) => break Err(()),
                                Err(STATE_UNINIT) => break Err(()),
                                Err(STATE_MID_UNINIT) => continue,
                                _ => unreachable!(),
                            }
                        }
                    })
                    .collect::<Result<(), ()>>() {
                    self
                        .reset_state
                        .compare_exchange(STATE_MID_UNINIT, STATE_INIT, Ordering::Relaxed, Ordering::Relaxed)
                        .map_err(|_| ())
                        .unwrap();
                    return Err(());
                }
                break;
            }

            self.indices.reset();

            #[cfg(debug_assertions)]
            self
                .reset_state
                .compare_exchange(STATE_MID_UNINIT, STATE_INIT, Ordering::Release, Ordering::Relaxed)
                .map_err(|_| ())
                .unwrap();

            #[cfg(not(debug_assertions))]
            self.reset_state.store(STATE_INIT, Ordering::Relaxed);

            Ok(())
        } else {
            Err(())
        };

        res
    }
}

impl<T, const LEN: usize> Drop for Chunk<T, LEN> {
    fn drop(&mut self) {
        for (item, state) in self.chunk_data.iter() {
            match state.load(Ordering::Relaxed) {
                STATE_FINISHED | STATE_UNINIT => {},
                STATE_MID_UNINIT | STATE_MID_INIT => unreachable!(),
                STATE_INIT => unsafe {
                    // Sanity check, as _ casts will only change one aspect of a pointer, type or mutability,
                    // hence, if we specify that we're changing mutability and expecting T as the type then
                    // it'll break if we aren't already *const T.
                    std::ptr::drop_in_place::<T>((&*item.get()).as_ptr() as *mut _);
                },
                _ => unreachable!(),
            }
        }
    }
}
