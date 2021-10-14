use arc_swap::{ArcSwapAny, RefCnt};
use ptr::addr_of_mut;
use std::alloc::Layout;
use std::cell::UnsafeCell;
use std::mem::{forget, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, AtomicUsize, Ordering};

const DEFAULT_LENGTH: usize = 64;

pub struct Queue<T> {
    data: ArcSwapAny<GrowableBundle<T>>,
    len: AtomicUsize,
    reorder_rights: AtomicUsize,
    generation: AtomicUsize,
}

unsafe impl<T: Send> Sync for Queue<T> {}
unsafe impl<T: Send> Send for Queue<T> {}

impl<T> Queue<T> {
    pub fn new() -> Self {
        Self {
            data: ArcSwapAny::new(GrowableBundle::new(1)),
            len: AtomicUsize::new(0),
            reorder_rights: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
        }
    }

    pub fn push(&self, mut val: T) {
        self.len.fetch_add(1, Ordering::SeqCst);
        loop {
            let generation = self.generation.load(Ordering::SeqCst);
            let ptr = self.data.load();
            let push_chunk = ptr.push_chunk();
            let slice = ptr.slice();

            let current_push_chunk = push_chunk.load(Ordering::SeqCst);

            for (idx, chunk) in slice[current_push_chunk..].iter().enumerate() {
                let idx = idx + current_push_chunk + 1;
                match unsafe { (**chunk.get()).push(val) } {
                    Err(x) => {
                        val = x;
                        if unsafe { (**chunk.get()).is_full() } {
                            push_chunk.fetch_max(idx, Ordering::SeqCst);
                        }
                    }
                    _ => return,
                }
            }

            if self.reorder_rights.fetch_add(1, Ordering::Acquire) == 0 {
                if self.generation.load(Ordering::SeqCst) == generation {
                    let new = if ptr.pop_chunk().load(Ordering::SeqCst) == 0 {
                        ptr.add_chunks(4, generation)
                    } else {
                        ptr.reorder_chunks().unwrap()
                    };

                    assert!(std::ptr::eq(ptr.ptr as _, self.data.swap(new).ptr as _));
                    self.generation.fetch_add(1, Ordering::SeqCst);
                }
            }
            self.reorder_rights.fetch_sub(1, Ordering::Release);
            while self.reorder_rights.load(Ordering::SeqCst) != 0 {}
        }
    }

    pub fn pop(&self) -> Option<T> {
        let mut ptr = self.data.load();

        loop {
            let current_chunk = ptr.pop_chunk();
            let slice = ptr.slice();

            let current_pop_chunk = current_chunk.load(Ordering::SeqCst);

            for (idx, chunk) in slice[current_pop_chunk..].iter().enumerate() {
                let idx = idx + current_pop_chunk + 1;
                match unsafe { (**chunk.get()).pop() } {
                    Ok(Some(x)) => {
                        self.len.fetch_sub(1, Ordering::SeqCst);
                        return Some(x);
                    }
                    Ok(None) =>
                        if self.len.load(Ordering::SeqCst) > 0 {
                            break;
                        } else {
                            return None;
                        }
                    ,
                    Err(()) => {
                        current_chunk.fetch_max(idx, Ordering::SeqCst);
                    },
                }
            }

            while self.reorder_rights.load(Ordering::SeqCst) != 0 {}
            let new = self.data.load();
            if std::ptr::eq(new.ptr, ptr.ptr) {
                break;
            } else {
                ptr = new;
            }
        }

        None
    }

    pub fn pop_weak(&self) -> Option<T> {
        let ptr = self.data.load();

        let pop_chunk = ptr.pop_chunk();
        let slice = ptr.slice();

        let current_pop_chunk = pop_chunk.load(Ordering::SeqCst);

        for (idx, chunk) in slice[current_pop_chunk..].iter().enumerate() {
            let idx = idx + current_pop_chunk + 1;
            match unsafe { (**chunk.get()).pop() } {
                Ok(Some(x)) => {
                    self.len.fetch_sub(1, Ordering::SeqCst);
                    return Some(x);
                }
                Ok(None) => return None,
                Err(()) => {
                    pop_chunk.fetch_max(idx, Ordering::SeqCst);
                },
            }
        }

        None
    }
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
                ptr::write(ptr, UCChunk::new(0));
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

    pub fn add_chunks(&self, chunks: usize, generation: usize) -> Self {
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
                ptr::write(ptr, UCChunk::new(generation + 1));
            }
        }

        Self { ptr: new }
    }

    pub fn reorder_chunks(&self) -> Result<Self, ()> {
        let copy = self.add_chunks(0, 0);
        let pop_chunk = self.pop_chunk().load(Ordering::SeqCst);
        let push_chunk = self.push_chunk().load(Ordering::SeqCst);
        let mut to_end: Vec<*mut Chunk<_, DEFAULT_LENGTH>> = Vec::with_capacity(pop_chunk);
        let mut correct = Vec::with_capacity(push_chunk - pop_chunk);

        for chunk in self.slice() {
            match unsafe { (**chunk.get()).try_reset() } {
                Ok(()) => {
                    to_end.push(unsafe { *chunk.get() });
                },
                Err(()) => {
                    correct.push(unsafe { *chunk.get() });
                }
            }
        }

        let correct_len = correct.len();

        correct
            .into_iter()
            .chain(to_end.into_iter())
            .zip(copy.slice())
            .for_each(|(src, dst)| unsafe {
                ptr::write(dst.get(), src);
            });

        copy.pop_chunk().store(0, Ordering::SeqCst);
        copy.push_chunk().store(correct_len, Ordering::SeqCst);

        Ok(copy)
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
                        STATE_UNINIT | STATE_FINISHED => {}
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
    pub fn new(gen: usize) -> Self {
        Self {
            inner: UnsafeCell::new(Box::into_raw(Box::new(Chunk::new(gen)))),
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
                    if end == LEN as u32 {
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

    pub fn pop(&self) -> Result<Option<usize>, ()> {
        match self
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

        start == LEN as u32
    }

    pub fn is_new(&self) -> bool {
        self
            .indices
            .load(Ordering::SeqCst)
            == 0
    }

    pub fn is_full(&self) -> bool {
        self
            .indices
            .load(Ordering::SeqCst)
            >> 16
            == LEN as u32
    }

    pub fn reset(&self) {
        self.indices.store(0, Ordering::SeqCst);
    }
}

const STATE_UNINIT: u8 = 0b000;
const STATE_MID_INIT: u8 = 0b001;
const STATE_INIT: u8 = 0b010;
const STATE_MID_UNINIT: u8 = 0b011;
const STATE_FINISHED: u8 = 0b100;

// SAFETY: The align of this struct may not change.
pub struct Chunk<T, const LEN: usize> {
    indices: ChunkIndices<LEN>,
    reset_state: AtomicU8,
    generation: AtomicUsize,
    chunk_data: [(UnsafeCell<MaybeUninit<T>>, AtomicU8); LEN],
}

unsafe impl<T: Send, const LEN: usize> Send for Chunk<T, LEN> {}
unsafe impl<T: Send, const LEN: usize> Sync for Chunk<T, LEN> {}

impl<T, const LEN: usize> Default for Chunk<T, LEN> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<T, const LEN: usize> Chunk<T, LEN> {
    pub fn new(gen: usize) -> Self {
        assert!(LEN <= u16::MAX as usize);
        Self {
            indices: ChunkIndices::new(),
            reset_state: AtomicU8::new(STATE_INIT),
            generation: AtomicUsize::new(gen),
            chunk_data: [(); LEN].map(|_| {
                (
                    UnsafeCell::new(MaybeUninit::uninit()),
                    AtomicU8::new(STATE_UNINIT),
                )
            }),
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
                    Ordering::SeqCst,
                    Ordering::SeqCst
                ) {
                    Ok(_) => {
                        let val = Some(unsafe {
                            std::ptr::read((*cell.get()).as_ptr())
                        });
                        state
                            .compare_exchange(
                                STATE_MID_UNINIT,
                                STATE_FINISHED,
                                Ordering::SeqCst,
                                Ordering::SeqCst,
                            )
                            .unwrap();
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
            match state.compare_exchange(STATE_UNINIT, STATE_MID_INIT, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => {
                    // self.indices.push().unwrap();
                    unsafe { std::ptr::write(cell.get(), MaybeUninit::new(val)) };
                    state.store(STATE_INIT, Ordering::SeqCst);
                    return Ok(());
                },
                _ => (),
            }
        }

        Err(val)
    }

    #[inline(never)]
    pub fn is_finished(&self) -> bool {
        self.indices.is_finished()
    }

    pub fn is_full(&self) -> bool {
        self.indices.is_full()
    }

    pub fn try_reset(&self) -> Result<(), ()> {
        if self.is_finished() {
            self.generation.fetch_add(1, Ordering::SeqCst);
            self
                .reset_state
                .compare_exchange(STATE_INIT, STATE_MID_UNINIT, Ordering::SeqCst, Ordering::Relaxed)
                .map_err(|_| ())?;

            loop {
                if let Err(()) = self.chunk_data
                    .iter()
                    .map(|(_, flag)| {
                        loop {
                            match flag
                                .compare_exchange(STATE_FINISHED, STATE_UNINIT, Ordering::SeqCst, Ordering::SeqCst) {
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
                        .compare_exchange(STATE_MID_UNINIT, STATE_INIT, Ordering::SeqCst, Ordering::Relaxed)
                        .map_err(|_| ())
                        .unwrap();
                    return Err(());
                }
                break;
            }

            self.indices.reset();

            self
                .reset_state
                .compare_exchange(STATE_MID_UNINIT, STATE_INIT, Ordering::SeqCst, Ordering::Relaxed)
                .map_err(|_| ())
                .unwrap();

            Ok(())
        } else {
            Err(())
        }
    }

    pub fn is_new(&self) -> bool {
        self.reset_state
            .load(Ordering::SeqCst)
            == STATE_INIT
            &&
            self
                .indices
                .is_new()
            &&
            self
                .chunk_data
                .iter()
                .all(|(_, state)| state.load(Ordering::SeqCst) == STATE_UNINIT)
    }

    pub fn update_generation(&self) -> usize {
        self.generation.fetch_add(1, Ordering::SeqCst)
    }

    pub fn generation(&self) -> usize {
        self.generation.load(Ordering::SeqCst)
    }
}
