use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU32, AtomicU8, AtomicUsize, Ordering, AtomicPtr};

const CHUNK_LEN: usize = 128;

pub struct Queue<T> {
    len: AtomicUsize,
    push: AtomicPtr<Chunk<T, CHUNK_LEN>>,
    pop: AtomicPtr<Chunk<T, CHUNK_LEN>>,

    // #[cfg(debug_assertions)]
    // pub chunks: AtomicUsize,
}

impl<T> Queue<T> {
    pub fn new() -> Self {
        let chunk = Box::into_raw(Box::new(Chunk::new()));

        unsafe {
            (&*chunk).next.store(chunk, Ordering::Relaxed);
            #[cfg(debug_assertions)]
            (&*chunk).prev.store(chunk, Ordering::Relaxed);
        }

        Self {
            len: AtomicUsize::new(0),
            push: AtomicPtr::new(chunk),
            pop: AtomicPtr::new(chunk),

            // #[cfg(debug_assertions)]
            // chunks: AtomicUsize::new(1),
        }
    }

    pub fn pop(&self) -> Option<T> {
        let mut len = self.len.load(Ordering::Relaxed);

        loop {
            if len == 0 {
                return None;
            }

            if let Err(new) = self.len.compare_exchange(
                len,
                len - 1,
                Ordering::Release,
                Ordering::Relaxed
            ) {
                len = new;
            } else {
                break;
            }
        }

        let mut this_chunk = self.pop.load(Ordering::Relaxed);

        let mut original_chunk = this_chunk;

        debug_assert!(!this_chunk.is_null());

        let val = unsafe {
            let mut result = (&*this_chunk).pop();

            while let Err(_) | Ok(None) = result {
                let _ = (&*this_chunk).try_reset();

                let mut new_chunk = (&*this_chunk).next.load(Ordering::Relaxed);

                while new_chunk.is_null() {
                    new_chunk = (&*this_chunk).next.load(Ordering::Relaxed);
                }
                this_chunk = new_chunk;

                let _ = self
                    .pop
                    .compare_exchange(
                        original_chunk,
                        this_chunk,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    );

                original_chunk = this_chunk;

                result = (&*this_chunk).pop();
            }

            result.unwrap().unwrap()
        };

        Some(val)
    }

    pub fn push(&self, mut val: T) {
        self.len.fetch_add(1, Ordering::Acquire);

        let mut this_chunk = self.push.load(Ordering::Relaxed);

        let mut original_chunk = this_chunk;

        'outer_loop: loop {
            match unsafe { (&*this_chunk).push(val) } {
                Ok(()) => break 'outer_loop,
                Err(v) => {
                    val = v;

                    let next_ptr = unsafe {
                        let mut next_ptr = (&*this_chunk).next.load(Ordering::Relaxed);

                        while next_ptr.is_null() {
                            next_ptr = (&*this_chunk).next.load(Ordering::Relaxed);
                        }
                        next_ptr
                    };

                    if unsafe { (&*next_ptr).is_full() } {
                        unsafe {
                            match (&*this_chunk).next
                                .compare_exchange(
                                    next_ptr,
                                    null_mut(),
                                    Ordering::Relaxed,
                                    Ordering::Relaxed
                                ) {
                                Ok(_) => {
                                    let new_chunk = Chunk::new();
                                    // Should never fail given we just constructed a new, empty chunk.
                                    let _ = new_chunk.push(val);
                                    let new_chunk = Box::new(new_chunk);
                                    new_chunk.next.store(next_ptr, Ordering::Relaxed);
                                    // #[cfg(debug_assertions)]
                                    // self.chunks.fetch_add(1, Ordering::Relaxed);
                                    #[cfg(debug_assertions)]
                                    (&*this_chunk)
                                        .next
                                        .compare_exchange(
                                            null_mut(),
                                            Box::into_raw(new_chunk),
                                            Ordering::Relaxed,
                                            Ordering::Relaxed,
                                        )
                                        .unwrap();

                                    #[cfg(not(debug_assertions))]
                                    (&*this_chunk)
                                        .next
                                        .store(Box::into_raw(new_chunk), Ordering::Release);

                                    break 'outer_loop;
                                },
                                Err(mut new) => {
                                    while new.is_null() {
                                        new = (&*this_chunk).next.load(Ordering::Relaxed);
                                    }

                                    continue 'outer_loop;
                                }
                            }
                        }
                    } else {
                        this_chunk = next_ptr;

                        let _ = self
                            .push
                            .compare_exchange(
                                original_chunk,
                                this_chunk,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            );

                        original_chunk = this_chunk;
                    }
                }
            }
        }
    }
}

impl<T> Drop for Queue<T> {
    fn drop(&mut self) {
        let first = self.push.load(Ordering::Relaxed);

        if first.is_null() {
            return;
        }

        unsafe {
            let mut this = first;
            let mut next = (&*first).next.load(Ordering::Relaxed);
            loop {
                drop(Box::from_raw(this));

                if std::ptr::eq(next, first) {
                    break;
                }

                #[cfg(debug_assertions)]
                if next.is_null() {
                    panic!("Next pointer is null when dropping!");
                }

                this = next;
                next = (&*next).next.load(Ordering::Relaxed);
            }
        }

        self.push.store(null_mut(), Ordering::Relaxed);
        self.pop.store(null_mut(), Ordering::Relaxed);
    }
}

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
    next: AtomicPtr<Chunk<T, LEN>>,
    // #[cfg(debug_assertions)]
    // prev: AtomicPtr<Chunk<T, LEN>>,
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
            // #[cfg(debug_assertions)]
            // prev: AtomicPtr::new(null_mut()),
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
        for (cell, state) in &self.chunk_data[pop_idx..] {
            loop {
                match state.compare_exchange(
                    STATE_INIT,
                    STATE_MID_UNINIT,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        let val = Some(unsafe {
                            std::ptr::read((*cell.get()).as_ptr())
                        });
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
                        state.store(STATE_FINISHED, Ordering::Relaxed);

                        return Ok(val);
                    },
                    Err(STATE_MID_INIT) => continue,
                    Err(STATE_UNINIT) => continue,
                    _ => break,
                }
            }
        }

        unreachable!()
    }

    pub fn push(&self, val: T) -> Result<(), T> {
        // let push_idx = self.indices.push_idx();
        let push_idx = if let Some(x) = self.indices.push() {
            x
        } else {
            return Err(val);
        };

        for (cell, state) in &self.chunk_data[push_idx..] {
            match state.compare_exchange(STATE_UNINIT, STATE_MID_INIT, Ordering::Acquire, Ordering::Relaxed) {
                Ok(_) => {
                    // self.indices.push().unwrap();
                    unsafe { std::ptr::write(cell.get(), MaybeUninit::new(val)) };
                    state.store(STATE_INIT, Ordering::Release);
                    return Ok(());
                },
                _ => (),
            }
        }

        Err(val)
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
