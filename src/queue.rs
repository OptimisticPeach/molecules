use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use crate::chunk::Chunk;

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

        #[cfg(debug_assertions)]
        self.push.store(null_mut(), Ordering::Relaxed);
        #[cfg(debug_assertions)]
        self.pop.store(null_mut(), Ordering::Relaxed);
    }
}
