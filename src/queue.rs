use arc_swap::{ArcSwapAny, RefCnt};
use ptr::addr_of_mut;
use std::alloc::Layout;
use std::cell::UnsafeCell;
use std::mem::{forget, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, AtomicUsize, Ordering};

const DEFAULT_LENGTH: usize = 64;

#[cold]
fn cold() {}

pub struct Queue<T> {
    data: ArcSwapAny<GrowableBundle<T>>,
    len: AtomicUsize,
}

unsafe impl<T: Send> Sync for Queue<T> {}
unsafe impl<T: Send> Send for Queue<T> {}

impl<T> Queue<T> {
    pub fn new() -> Self {
        Self {
            data: ArcSwapAny::new(GrowableBundle::new(1)),
            len: AtomicUsize::new(0),
        }
    }

    pub fn push(&self, mut val: T) {
        self.len.fetch_add(1, Ordering::SeqCst);
        loop {
            let ptr = self.data.load();
            let current_chunk = ptr.push_chunk();
            let slice = ptr.slice();

            let current_push_chunk = current_chunk.load(Ordering::SeqCst);

            for (idx, chunk) in slice[current_push_chunk..].iter().enumerate() {
                let idx = idx + current_push_chunk;
                match unsafe { (**chunk.get()).push(val) } {
                    Err(x) => {
                        val = x;
                        current_chunk.store(idx, Ordering::SeqCst);
                    }
                    _ => return,
                }
            }

            if ptr.pop_chunk().load(Ordering::SeqCst) == 0 {
                cold();
                let new = ptr.add_chunks(4);
                let old = self.data.compare_and_swap(&ptr, new.clone());
                if !std::ptr::eq(old.ptr as _, ptr.ptr as _) {
                    unsafe {
                        new.invalidate_last(4);
                    }
                }
            } else {
                let new = ptr.reorder_chunks(0);
                self.data.compare_and_swap(&ptr, new.clone());
            }
        }
    }

    pub fn pop(&self) -> Option<T> {
        self.len.fetch_sub(1, Ordering::SeqCst);
        let ptr = self.data.load();
        let current_chunk = ptr.pop_chunk();
        let slice = ptr.slice();

        let current_pop_chunk = current_chunk.load(Ordering::SeqCst);

        for (idx, chunk) in slice[current_pop_chunk..].iter().enumerate() {
            let idx = idx + current_pop_chunk;
            match unsafe { (**chunk.get()).pop() } {
                Some(x) => return Some(x),
                None => if unsafe { (**chunk.get()).is_finished() } {
                    current_chunk.store(idx, Ordering::SeqCst);
                } else {
                    return None;
                }
            }
        }

        None
    }

    // pub fn len(&self) -> usize {
    //     self.len.load(Ordering::Relaxed)
    // }

    // pub fn cap(&self) -> usize {
    //     self.data.load().total_len()
    // }
}

impl<T> Drop for Queue<T> {
    fn drop(&mut self) {
        unsafe { self.data.load().set_drop_flag() }
    }
}

struct GrowableBundle<T> {
    ptr: *mut Growable<T>,
}

impl<T> Clone for GrowableBundle<T> {
    fn clone(&self) -> Self {
        self.atomic_count().fetch_add(1, Ordering::SeqCst);

        Self { ptr: self.ptr }
    }
}

impl<T> Drop for GrowableBundle<T> {
    fn drop(&mut self) {
        let old_refcount = self.atomic_count().fetch_sub(1, Ordering::SeqCst);
        if old_refcount == 1 {
            if unsafe { (*self.ptr).tail.drop_flag.load(Ordering::SeqCst) } {
                unsafe { self.deallocate_recursive() }
            } else {
                unsafe {
                    self.deallocate();
                }
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
            let offset = (*growable).tail.offset;

            let tail = GrowableTail {
                chunks,
                offset,
                atomic_count: AtomicUsize::new(1),
                current_chunk_push: AtomicUsize::new(0),
                current_chunk_pop: AtomicUsize::new(0),
                drop_flag: AtomicBool::new(false),
            };

            ptr::write(addr_of_mut!((*growable).tail), tail);

            offset
        };

        let slice_ptr = unsafe { growable.cast::<u8>().add(offset).cast::<UCChunk<T>>() };

        for ptr in (0..chunks).map(|x| unsafe { slice_ptr.add(x) }) {
            unsafe {
                ptr::write(ptr, UCChunk::new());
            }
        }

        Self { ptr: growable }
    }

    pub unsafe fn invalidate_last(mut self, offending: usize) {
        assert_eq!(self.atomic_count().fetch_sub(1, Ordering::SeqCst), 1);
        let slice = self.slice();
        for slot in &slice[slice.len() - offending..] {
            // Drop it.
            Box::from_raw(*slot.get());
        }

        self.deallocate();
        forget(self);
    }

    pub fn add_chunks(&self, chunks: usize) -> Self {
        let len = self.chunks() + chunks;
        let new = Self::alloc(len);
        let push_chunk = self.push_chunk().load(Ordering::SeqCst);
        let pop_chunk = self.pop_chunk().load(Ordering::SeqCst);

        unsafe {
            let offset = (*new).tail.offset;

            let tail = GrowableTail {
                chunks: len,
                offset,
                atomic_count: AtomicUsize::new(1),
                current_chunk_push: AtomicUsize::new(push_chunk),
                current_chunk_pop: AtomicUsize::new(pop_chunk),
                drop_flag: AtomicBool::new(false),
            };

            ptr::write(addr_of_mut!((*new).tail), tail);
        }

        let slice_ptr = unsafe {
            new.cast::<u8>()
                .add((*new).tail.offset)
                .cast::<UCChunk<T>>()
        };

        unsafe {
            // SAFETY: `self.slice`'s ptrs are read-only currently.
            ptr::copy_nonoverlapping(self.slice().as_ptr(), slice_ptr, self.chunks());
        }

        if chunks == 0 {
            return Self { ptr: new };
        }

        unsafe {
            for ptr in (self.chunks()..len).map(|x| slice_ptr.add(x)) {
                ptr::write(ptr, UCChunk::new());
            }
        }

        Self { ptr: new }
    }

    pub fn reorder_chunks(&self, extra: usize) -> Self {
        let copy = self.add_chunks(extra);
        let pop_chunk = self.pop_chunk().load(Ordering::SeqCst);
        let push_chunk = self.push_chunk().load(Ordering::SeqCst);
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
            .for_each(|(src, dst)| unsafe {
                ptr::write(dst.get(), src);
            });

        copy.pop_chunk().store(0, Ordering::SeqCst);
        copy.push_chunk().store(correct_len, Ordering::SeqCst);

        copy
    }

    #[inline]
    pub fn chunks(&self) -> usize {
        unsafe { (*self.ptr).tail.chunks }
    }

    // #[inline]
    // pub fn total_len(&self) -> usize {
    //     unsafe { (*self.ptr).tail.total_len }
    // }

    #[inline]
    fn atomic_count(&self) -> &AtomicUsize {
        unsafe { &(*self.ptr).tail.atomic_count }
    }

    #[inline]
    pub fn push_chunk(&self) -> &AtomicUsize {
        unsafe { &(*self.ptr).tail.current_chunk_push }
    }

    #[inline]
    pub fn pop_chunk(&self) -> &AtomicUsize {
        unsafe { &(*self.ptr).tail.current_chunk_pop }
    }

    #[inline]
    pub fn slice(&self) -> &[UCChunk<T>] {
        unsafe {
            let offset = (*self.ptr).tail.offset;
            let ptr = self.ptr.cast::<u8>().add(offset).cast::<UCChunk<T>>();

            &*std::ptr::slice_from_raw_parts(ptr, self.chunks())
        }
    }

    #[inline]
    pub unsafe fn set_drop_flag(&self) {
        (*self.ptr).tail.drop_flag.store(true, Ordering::SeqCst);
    }

    fn alloc(len: usize) -> *mut Growable<T> {
        let (layout, start) = Layout::new::<GrowableTail>()
            .extend(Layout::array::<UCChunk<T>>(len).unwrap())
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
        assert_eq!(
            self.atomic_count().load(Ordering::SeqCst),
            0,
            "Internal error, dropping a growable without having all instances of it detached!"
        );
        let (layout, _) = Layout::new::<GrowableTail>()
            .extend(Layout::array::<UCChunk<T>>(len).unwrap())
            .unwrap();
        std::alloc::dealloc(self.ptr.cast(), layout.pad_to_align());
    }

    unsafe fn deallocate_recursive(&mut self) {
        assert_eq!(
            self.atomic_count().load(Ordering::SeqCst),
            0,
            "Internal error, dropping a growable without having all instances of it detached!"
        );

        self.slice().iter().for_each(|x| {
            unsafe {
                let boxed = Box::from_raw(*x.inner.get());

                for (item, state) in boxed.chunk_data {
                    match state.load(Ordering::SeqCst) {
                        STATE_UNINIT => {}
                        STATE_MID_INIT => panic!("This should never happen!"),
                        STATE_INIT => {
                            std::ptr::drop_in_place((*item.get()).as_ptr() as *mut T);
                        }
                        _ => unreachable!(),
                    }
                }
            }
        });

        self.deallocate();
    }
}

#[repr(C)]
struct GrowableTail {
    chunks: usize,
    // total_len: usize,
    offset: usize,
    atomic_count: AtomicUsize,
    current_chunk_push: AtomicUsize,
    current_chunk_pop: AtomicUsize,
    drop_flag: AtomicBool,
}

#[repr(C)]
struct Growable<T> {
    tail: GrowableTail,
    _phantom: [UCChunk<T>; 0],
}

#[repr(transparent)]
pub struct UCChunk<T> {
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
            inner: UnsafeCell::new(Box::into_raw(Box::new(Chunk::new()))),
        }
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
                Ordering::SeqCst,
                Ordering::SeqCst,
                |x| {
                    let end = x >> 16;
                    if end == LEN as _ {
                        return None;
                    }
                    Some((end + 1) << 16 | (x & 0xffff))
                }
            )
            .ok()
            .map(|x| (x >> 16) as _)
    }

    pub fn push_idx(&self) -> usize {
        self
            .indices
            .load(Ordering::SeqCst)
            as usize
            >> 16
    }

    pub fn pop(&self) -> Option<usize> {
        self
            .indices
            .fetch_update(
                Ordering::SeqCst,
                Ordering::SeqCst,
                |x| {
                    let start = (x & 0xffff) + 1;
                    let end = x >> 16;

                    if start > end || start > LEN as _ {
                        return None;
                    }

                    Some((end << 16) | start)
                }
            )
            .ok()
            .map(|x| (x & 0xffff) as usize)
    }

    pub fn pop_idx(&self) -> usize {
        self
            .indices
            .load(Ordering::SeqCst)
            as usize
            & 0xffff
    }

    pub fn is_finished(&self) -> bool {
        let state = self
            .indices
            .load(Ordering::SeqCst);

        let start = state & 0xffff;

        start == LEN as _
    }

    pub fn reset(&self) {
        self.indices.store(0, Ordering::SeqCst);
    }
}

const STATE_UNINIT: u8 = 0b00;
const STATE_MID_INIT: u8 = 0b01;
const STATE_INIT: u8 = 0b10;
const STATE_MID_UNINIT: u8 = 0b11;

// SAFETY: The align of this struct may not change.
pub struct Chunk<T, const LEN: usize> {
    indices: ChunkIndices<LEN>,
    chunk_data: [(UnsafeCell<MaybeUninit<T>>, AtomicU8); LEN],
}

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
            chunk_data: [(); LEN].map(|_| {
                (
                    UnsafeCell::new(MaybeUninit::uninit()),
                    AtomicU8::new(STATE_UNINIT),
                )
            }),
        }
    }

    pub fn pop(&self) -> Option<T> {
        let pop_idx = self.indices.pop_idx();
        for (cell, state) in &self.chunk_data[pop_idx..] {
            match state.compare_exchange(
                STATE_INIT,
                STATE_MID_UNINIT,
                Ordering::SeqCst,
                Ordering::SeqCst
            ) {
                Ok(_) => {
                    self.indices.pop().unwrap();
                    let val = Some(unsafe {
                        std::ptr::read((*cell.get()).as_ptr())
                    });
                    state
                        .compare_exchange(
                            STATE_MID_UNINIT,
                            STATE_UNINIT,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        )
                        .unwrap();
                    return val;
                },
                _ => continue,
            }
        }

        None
    }

    pub fn push(&self, val: T) -> Result<(), T> {
        let push_idx = self.indices.push_idx();
        if push_idx == LEN {
            return Err(val);
        }

        for (cell, state) in &self.chunk_data[push_idx..] {
            if state.compare_exchange(STATE_UNINIT, STATE_MID_INIT, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                self.indices.push().unwrap();
                unsafe { std::ptr::write(cell.get(), MaybeUninit::new(val)) };
                state.store(STATE_INIT, Ordering::SeqCst);
                return Ok(());
            }
        }

        Err(val)
    }

    pub fn is_finished(&self) -> bool {
        self.indices.is_finished()
    }

    pub fn try_reset(&self) -> Result<(), ()> {
        self.chunk_data
            .iter()
            .for_each(|(_, flag)| {
                loop {
                    match flag
                        .load(Ordering::SeqCst) {
                        STATE_UNINIT => break,
                        STATE_MID_INIT | STATE_INIT => unreachable!(),
                        STATE_MID_UNINIT => continue,
                        _ => unreachable!(),
                    }
                }
            });
        self.indices.reset();

        Ok(())
    }
}
