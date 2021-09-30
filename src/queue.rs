use arc_swap::{ArcSwapAny, RefCnt};
use std::alloc::Layout;
use std::cell::UnsafeCell;
use std::mem::{forget, MaybeUninit};
use std::ptr;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use ptr::addr_of_mut;
use std::ops::{Deref, DerefMut};

const DEFAULT_LENGTH: usize = 64;

pub struct Queue<T> {
    data: ArcSwapAny<GrowableBundle<T>>,
}

impl<T> Queue<T> {
    pub fn new() -> Self {
        Self {
            data: ArcSwapAny::new(GrowableBundle::new(1)),
        }
    }

    pub fn push(&self, mut val: T) {
        loop {
            let ptr = self.data.load();
            let current_chunk = ptr.push_chunk();
            let slice = ptr.slice();

            let current_push_chunk = current_chunk.load(Ordering::Relaxed);

            for (idx, chunk) in slice[current_push_chunk..].iter().enumerate() {
                let idx = idx + current_push_chunk;
                match unsafe {
                    (**chunk.get()).push(val)
                } {
                    Err(x) => {
                        val = x;
                        current_chunk.store(idx, Ordering::Relaxed);
                    },
                    _ => return,
                }
            }

            if current_push_chunk == 0 {
                let new = ptr.add_chunks(4);
                let old = self.data.compare_and_swap(&ptr, new.clone());
                if !std::ptr::eq(old.ptr as _, ptr.ptr as _) {
                    unsafe {
                        new.invalidate_last(4);
                    }
                }
            } else {
                let new = ptr.reorder_chunks(2);
                let old = self.data.compare_and_swap(&ptr, new.clone());
                if !std::ptr::eq(old.ptr as _, ptr.ptr as _) {
                    unsafe {
                        new.invalidate_last(2);
                    }
                }
            }
        }
    }

    pub fn pop(&self) -> Option<T> {
        let ptr = self.data.load();
        let current_chunk = ptr.pop_chunk();
        let slice = ptr.slice();

        let current_pop_chunk = current_chunk.load(Ordering::Relaxed);

        for (idx, chunk) in slice[current_pop_chunk..].iter().enumerate() {
            let idx = idx + current_pop_chunk;
            match unsafe {
                (**chunk.get()).pop()
            } {
                Ok(x) => return x,
                Err(_) => current_chunk.store(idx, Ordering::Relaxed),
            }
        }

        None
    }
}

impl<T> Drop for Queue<T> {
    fn drop(&mut self) {
        unsafe {
            // todo?
        }
    }
}

struct GrowableBundle<T> {
    ptr: *mut Growable<T>,
}

impl<T> Clone for GrowableBundle<T> {
    fn clone(&self) -> Self {
        self.atomic_count().fetch_add(1, Ordering::Relaxed);

        Self { ptr: self.ptr }
    }
}

impl<T> Drop for GrowableBundle<T> {
    fn drop(&mut self) {
        let old_refcount = self.atomic_count().fetch_sub(1, Ordering::Relaxed);
        if old_refcount == 1 {
            unsafe {
                self.deallocate();
            }
        }
    }
}

unsafe impl<T> RefCnt for GrowableBundle<T> {
    type Base = Growable<T>;

    fn into_ptr(me: Self) -> *mut Self::Base {
        let ptr = me.ptr;
        forget(me);
        ptr
    }

    fn as_ptr(me: &Self) -> *mut Self::Base {
        me.ptr
    }

    unsafe fn from_ptr(ptr: *const Self::Base) -> Self {
        Self { ptr: ptr as _ }
    }
}

impl<T> GrowableBundle<T> {
    pub fn new(chunks: usize) -> Self {
        assert_ne!(chunks, 0, "Cannot allocate a zero-length growable buffer.");
        let growable = Self::alloc(chunks);
        let offset = unsafe {
            ptr::write(addr_of_mut!((*growable).tail.chunks), chunks);
            ptr::write(addr_of_mut!((*growable).tail.total_len), DEFAULT_LENGTH * chunks);
            ptr::write(addr_of_mut!((*growable).tail.atomic_count), AtomicUsize::new(1));
            ptr::write(addr_of_mut!((*growable).tail.current_chunk_push), AtomicUsize::new(0));
            ptr::write(addr_of_mut!((*growable).tail.current_chunk_pop), AtomicUsize::new(0));

            (*growable).tail.offset
        };

        let slice_ptr = unsafe {
            growable
                .cast::<u8>()
                .add(offset)
                .cast::<UCChunk<T>>()
        };

        for ptr in (0..chunks).map(|x| unsafe { slice_ptr.add(x) }) {
            unsafe {
                ptr::write(
                    ptr,
                    UCChunk::new(),
                );
            }
        }

        Self {
            ptr: growable
        }
    }

    pub unsafe fn invalidate_last(mut self, offending: usize) {
        assert_eq!(self.atomic_count().fetch_sub(1, Ordering::Relaxed), 1);
        let slice = self.slice();
        for slot in &slice[slice.len() - offending..] {
            // Drop it.
            Box::from_raw(*slot.get());
        }

        self.deallocate();
        forget(self);
    }

    pub unsafe fn drop_all(mut self) {
        assert_eq!(self.atomic_count().fetch_sub(1, Ordering::Relaxed), 1);
        self.deallocate_recursive();
        forget(self);
    }

    pub fn add_chunks(&self, chunks: usize) -> Self {
        let len = self.chunks() + chunks;
        let new = Self::alloc(len);
        let push_chunk = self.push_chunk().load(Ordering::Relaxed);
        let pop_chunk = self.pop_chunk().load(Ordering::Relaxed);

        unsafe {
            ptr::write(addr_of_mut!((*new).tail.chunks), len);
            ptr::write(addr_of_mut!((*new).tail.current_chunk_push), AtomicUsize::new(push_chunk));
            ptr::write(addr_of_mut!((*new).tail.current_chunk_pop), AtomicUsize::new(pop_chunk));
            ptr::write(addr_of_mut!((*new).tail.total_len), DEFAULT_LENGTH * len);
        }

        let slice_ptr = unsafe {
            new
                .cast::<u8>()
                .add((*new).tail.offset)
                .cast::<UCChunk<T>>()
        };

        unsafe {
            // SAFETY: `self.slice`'s ptrs are read-only currently.
            ptr::copy_nonoverlapping(
                self.slice().as_ptr(),
                slice_ptr,
                self.chunks(),
            );
        }

        unsafe {
            for ptr in (self.chunks()..len).map(|x| slice_ptr.add(x)) {
                ptr::write(
                    ptr,
                    UCChunk::new(),
                );
            }
        }

         Self {
             ptr: new,
         }
    }

    pub fn reorder_chunks(&self, extra: usize) -> Self {
        let copy = self.add_chunks(extra);
        let pop_chunk = self.pop_chunk().load(Ordering::Relaxed);
        let push_chunk = self.push_chunk().load(Ordering::Relaxed);
        let mut to_end = Vec::with_capacity(pop_chunk);
        let mut correct = Vec::with_capacity(push_chunk - pop_chunk);

        for chunk in self.slice() {
            unsafe {
                if (**chunk.get()).is_finished() {
                    to_end.push(ptr::read(chunk.get()));
                } else {
                    correct.push(ptr::read(chunk.get()));
                }
            }
        }

        let correct_len = correct.len();

        correct
            .into_iter()
            .chain(to_end.into_iter().inspect(|x| unsafe { (**x).reset() }))
            .zip(copy.slice())
            .for_each(|(src, dst)| {
                unsafe {
                    ptr::write(dst.get(), src);
                }
            });

        copy.pop_chunk().store(0, Ordering::Relaxed);
        copy.push_chunk().store(correct_len, Ordering::Relaxed);

        copy
    }

    #[inline]
    pub fn chunks(&self) -> usize {
        unsafe {
            (*self.ptr).tail.chunks
        }
    }

    #[inline]
    pub fn total_len(&self) -> usize {
        unsafe {
            (*self.ptr).tail.total_len
        }
    }

    #[inline]
    fn atomic_count(&self) -> &AtomicUsize {
        unsafe {
            &(*self.ptr).tail.atomic_count
        }
    }

    #[inline]
    pub fn push_chunk(&self) -> &AtomicUsize {
        unsafe {
            &(*self.ptr).tail.current_chunk_push
        }
    }

    #[inline]
    pub fn pop_chunk(&self) -> &AtomicUsize {
        unsafe {
            &(*self.ptr).tail.current_chunk_pop
        }
    }

    #[inline]
    pub fn slice(&self) -> &[UCChunk<T>] {
        unsafe {
            let offset = (*self.ptr).tail.offset;
            let ptr = self
                .ptr
                .cast::<u8>()
                .add(offset)
                .cast::<UCChunk<T>>();

            &*std::ptr::slice_from_raw_parts(ptr, self.chunks())
        }
    }

    fn alloc(len: usize) -> *mut Growable<T> {
        let (layout, start) = Layout::new::<GrowableTail>()
            .extend(
                Layout::array::<UCChunk<T>>(len)
                    .unwrap()
            )
            .unwrap();
        let memory = unsafe { std::alloc::alloc(layout.pad_to_align()) };

        if memory.is_null() {
            panic!("Failed to allocate memory!");
        }

        let memory = memory.cast::<Growable<T>>();

        unsafe {
            std::ptr::write(addr_of_mut!((*memory).tail.offset), start);
        }

        memory
    }

    unsafe fn deallocate(&mut self) {
        let len = self.chunks();
        assert_eq!(self.atomic_count().load(Ordering::Relaxed), 0, "Internal error, dropping a growable without having all instances of it detached!");
        let (layout, _) = Layout::new::<GrowableTail>()
            .extend(
                Layout::array::<UCChunk<T>>(len)
                    .unwrap()
            )
            .unwrap();
        std::alloc::dealloc(self.ptr.cast(), layout.pad_to_align());
    }

    unsafe fn deallocate_recursive(&mut self) {
        assert_eq!(self.atomic_count().load(Ordering::Relaxed), 0, "Internal error, dropping a growable without having all instances of it detached!");

        self
            .slice()
            .iter()
            .for_each(|x| drop(
                unsafe {
                    Box::from_raw(
                        *x.inner.get()
                    )
                }
            ));

        self.deallocate();
    }
}

#[repr(C)]
struct GrowableTail {
    chunks: usize,
    total_len: usize,
    offset: usize,
    atomic_count: AtomicUsize,
    current_chunk_push: AtomicUsize,
    current_chunk_pop: AtomicUsize,
}

#[repr(C)]
struct Growable<T> {
    tail: GrowableTail,
    _phantom: [UCChunk<T>; 0],
}

#[repr(transparent)]
struct UCChunk<T> {
    inner: UnsafeCell<*mut Chunk<T, DEFAULT_LENGTH>>,
}

impl<T> Deref for UCChunk<T> {
    type Target = UnsafeCell<*mut Chunk<T, DEFAULT_LENGTH>>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for UCChunk<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T> UCChunk<T> {
    pub fn new() -> Self {
        Self {
            inner: UnsafeCell::new(Box::into_raw(Box::new(Chunk::new())))
        }
    }
}

const STATE_UNINIT: u8 = 0b00;
const STATE_MID_INIT: u8 = 0b01;
const STATE_INIT: u8 = 0b11;

// SAFETY: The align of this struct may not change.
struct Chunk<T, const LEN: usize> {
    start: AtomicUsize,
    end: AtomicUsize,
    chunk_data: [(UnsafeCell<MaybeUninit<T>>, AtomicU8); LEN],
}

impl<T, const LEN: usize> Default for Chunk<T, LEN> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const LEN: usize> Chunk<T, LEN> {
    pub fn new() -> Self {
        Self {
            start: AtomicUsize::new(0),
            end: AtomicUsize::new(0),
            chunk_data: [(); LEN]
                .map(|_| {
                    (
                        UnsafeCell::new(MaybeUninit::uninit()),
                        AtomicU8::new(STATE_UNINIT)
                    )
                }),
        }
    }

    pub const fn len(&self) -> usize {
        LEN
    }

    pub fn pop(&self) -> Result<Option<T>, ()> {
        loop {
            let start = self.start.load(Ordering::Relaxed);
            if start == self.len() {
                return Err(());
            }
            if start == self.end.load(Ordering::Relaxed) {
                return Ok(None);
            }
            match self.start.compare_exchange(
                start,
                start + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => loop {
                    match self.chunk_data[start].1.compare_exchange(
                        STATE_INIT,
                        STATE_UNINIT,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            return Ok(Some(unsafe {
                                std::ptr::read((*self.chunk_data[start].0.get()).as_ptr())
                            }))
                        }
                        Err(STATE_UNINIT) => panic!("Invalid state reached!"),
                        Err(STATE_MID_INIT) => continue,
                        _ => unreachable!(),
                    }
                },
                Err(_) => continue,
            }
        }
    }

    pub fn push(&self, val: T) -> Result<(), T> {
        let end = self.end.load(Ordering::Relaxed);
        for cell in &self.chunk_data[end..] {
            match cell.1.compare_exchange(
                STATE_UNINIT,
                STATE_MID_INIT,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.end.fetch_add(1, Ordering::Relaxed);
                    unsafe { std::ptr::write(cell.0.get(), MaybeUninit::new(val)) };
                    cell.1.store(STATE_INIT, Ordering::Release);
                    return Ok(());
                }
                Err(STATE_INIT) => panic!("Invalid state reached!"),
                Err(STATE_MID_INIT) => continue,
                _ => unreachable!(),
            }
        }

        Err(val)
    }

    pub fn is_finished(&self) -> bool {
        self.start.load(Ordering::Relaxed) == self.len()
    }

    pub unsafe fn reset(&self) {
        self.start.store(0, Ordering::Relaxed);
        self.end.store(1, Ordering::Relaxed);
        self
            .chunk_data
            .iter()
            .for_each(|(_, flag)| flag.store(STATE_UNINIT, Ordering::Relaxed));
    }
}
