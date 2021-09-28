use arc_swap::{ArcSwapAny, RefCnt};
use std::alloc::Layout;
use std::cell::UnsafeCell;
use std::mem::{align_of, size_of, ManuallyDrop, MaybeUninit, forget};
use std::ptr;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use ptr::addr_of_mut;

const DEFAULT_LENGTH: usize = 64;

// todo: replace `Box` with `*mut _` since `Box` are mutually exclusive.
type UCChunk<T> = UnsafeCell<ManuallyDrop<Box<Chunk<T, DEFAULT_LENGTH>>>>;

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
                    chunk.get().as_ref().unwrap().push(val)
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
                self.data.compare_and_swap(ptr, new);
            } else {
                let new = ptr.reorder_chunks(2);
                self.data.compare_and_swap(ptr, new);
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
                chunk.get().as_ref().unwrap().pop()
            } {
                Ok(x) => return x,
                Err(_) => current_chunk.store(idx, Ordering::Relaxed),
            }
        }

        None
    }
}

struct GrowableBundle<T> {
    ptr: *mut Growable<T>,
}

impl<T> Clone for GrowableBundle<T> {
    fn clone(&self) -> Self {
        unsafe {
            atomic_count_of(self.ptr).as_ref().unwrap().fetch_add(1, Ordering::Relaxed);
        }

        Self { ptr: self.ptr }
    }
}

impl<T> Drop for GrowableBundle<T> {
    fn drop(&mut self) {
        let old_refcount = unsafe { atomic_count_of(self.ptr).as_ref().unwrap().fetch_sub(1, Ordering::Relaxed) };
        if old_refcount == 1 {
            unsafe {
                drop_growable(self.ptr);
            }
        }
    }
}

unsafe impl<T> RefCnt for GrowableBundle<T> {
    type Base = Growable<T>;

    fn into_ptr(me: Self) -> *mut Self::Base {
        me.ptr
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
        Self {
            // This comes with a refcount of 1.
            ptr: growable_new(chunks),
        }
    }

    pub unsafe fn invalidate_last(self, offending: usize) {
        assert_eq!(self.atomic_count().load(Ordering::Relaxed), 1);
        let slice = self.slice();
        for slot in &slice[slice.len() - offending..] {
            ManuallyDrop::drop(&mut *slot.get())
        }
    }

    pub unsafe fn drop_all(self) {
        assert_eq!(self.atomic_count().fetch_sub(1, Ordering::Relaxed), 1);
        drop_recursive_growable(self.ptr);
        forget(self);
    }

    pub fn add_chunks(&self, chunks: usize) -> Self {
        let ptr = unsafe {
            resize_growable(self.ptr, chunks)
        };

        Self {
            ptr,
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
                if chunk.get().as_ref().unwrap().is_finished() {
                    to_end.push(ptr::read(chunk.get()));
                } else {
                    correct.push(ptr::read(chunk.get()));
                }
            }
        }

        let correct_len = correct.len();

        correct
            .into_iter()
            .chain(to_end.into_iter())
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
            chunks_of(self.ptr)
        }
    }

    #[inline]
    pub fn total_len(&self) -> usize {
        unsafe {
            total_len_of(self.ptr)
        }
    }

    #[inline]
    pub fn atomic_count(&self) -> &AtomicUsize {
        unsafe {
            &*atomic_count_of(self.ptr)
        }
    }

    #[inline]
    pub fn push_chunk(&self) -> &AtomicUsize {
        unsafe {
            &*current_chunk_push_of(self.ptr)
        }
    }

    #[inline]
    pub fn pop_chunk(&self) -> &AtomicUsize {
        unsafe {
            &*current_chunk_pop_of(self.ptr)
        }
    }

    #[inline]
    pub fn slice(&self) -> &[UCChunk<T>] {
        unsafe {
            &*slice_of(self.ptr)
        }
    }
}

#[repr(C)]
struct GrowableTail {
    chunks: usize,
    total_len: usize,
    allocated_chunks: usize,
    atomic_count: AtomicUsize,
    current_chunk_push: AtomicUsize,
    current_chunk_pop: AtomicUsize,
}

struct Growable<T> {
    data: UCChunk<T>,
}

#[inline]
unsafe fn tail_of<T>(ptr: *mut Growable<T>) -> *mut GrowableTail {
    ptr.cast::<GrowableTail>().sub(1)
}

#[inline]
unsafe fn chunks_of<T>(ptr: *mut Growable<T>) -> usize {
    (*tail_of(ptr)).chunks
}

#[inline]
unsafe fn total_len_of<T>(ptr: *mut Growable<T>) -> usize {
    (*tail_of(ptr)).total_len
}

#[inline]
unsafe fn atomic_count_of<T>(ptr: *mut Growable<T>) -> *const AtomicUsize {
    ptr::addr_of!((*tail_of(ptr)).atomic_count)
}

#[inline]
unsafe fn current_chunk_push_of<T>(ptr: *mut Growable<T>) -> *const AtomicUsize {
    ptr::addr_of!((*tail_of(ptr)).current_chunk_push)
}

#[inline]
unsafe fn current_chunk_pop_of<T>(ptr: *mut Growable<T>) -> *const AtomicUsize {
    ptr::addr_of!((*tail_of(ptr)).current_chunk_pop)
}

#[inline]
unsafe fn slice_of<T>(ptr: *mut Growable<T>) -> *mut [UCChunk<T>] {
    let len = chunks_of(ptr);
    std::ptr::slice_from_raw_parts_mut(ptr.cast(), len)
}

fn alloc_growable<T>(len: usize) -> *mut Growable<T> {
    let align = align_of::<UCChunk<T>>();
    let tail_size = size_of::<GrowableTail>();
    let extra = tail_size / align + if tail_size % align == 0 { 0 } else { 1 };
    let total_len = extra + len;

    let layout = Layout::array::<UCChunk<T>>(total_len).unwrap();
    let memory = unsafe { std::alloc::alloc(layout) };

    if memory.is_null() {
        panic!("Failed to allocate memory!");
    }

    let ptr = unsafe { memory.cast::<UCChunk<T>>().add(extra).cast::<Growable<T>>() };

    unsafe {
        (*tail_of(ptr)).allocated_chunks = total_len;
    }

    ptr
}

fn growable_new<T>(chunks: usize) -> *mut Growable<T> {
    assert_ne!(chunks, 0, "Cannot allocate a zero-length growable buffer.");
    let growable = alloc_growable(chunks);
    unsafe {
        let tail = tail_of(growable);
        ptr::write(addr_of_mut!((*tail).chunks), chunks);
        ptr::write(addr_of_mut!((*tail).total_len), DEFAULT_LENGTH * chunks);
        ptr::write(addr_of_mut!((*tail).atomic_count), AtomicUsize::new(1));
        ptr::write(addr_of_mut!((*tail).current_chunk_push), AtomicUsize::new(0));
        ptr::write(addr_of_mut!((*tail).current_chunk_pop), AtomicUsize::new(0));
    }

    for ptr in (0..chunks).map(|x| unsafe { growable.cast::<UCChunk<T>>().add(x) }) {
        unsafe {
            ptr::write(
                ptr,
                Default::default(),
            );
        }
    }

    growable
}

/// # SAFETY
/// - `old` must be a valid (not dropped) and constructed from either
///   `growable_new` or `resize_growable`.
/// - The length of `old` must not be 0.
unsafe fn resize_growable<T>(old: *mut Growable<T>, add: usize) -> *mut Growable<T> {
    let old_len = chunks_of(old);
    let len = old_len + add;
    let growable = alloc_growable(len);
    let current_chunk_push = AtomicUsize::new((*current_chunk_push_of(old)).load(Ordering::Relaxed));
    let current_chunk_pop = AtomicUsize::new((*current_chunk_pop_of(old)).load(Ordering::Relaxed));

    let tail = tail_of(growable);
    ptr::write(addr_of_mut!((*tail).chunks), len);
    ptr::write(addr_of_mut!((*tail).total_len), DEFAULT_LENGTH * len);
    ptr::write(addr_of_mut!((*tail).atomic_count), AtomicUsize::new(1));
    ptr::write(addr_of_mut!((*tail).current_chunk_push), current_chunk_push);
    ptr::write(addr_of_mut!((*tail).current_chunk_pop), current_chunk_pop);

    ptr::copy_nonoverlapping(
        old.cast::<UCChunk<T>>(),
        growable.cast::<UCChunk<T>>(),
        old_len,
    );

    for ptr in (old_len..len).map(|x| unsafe { growable.cast::<UCChunk<T>>().add(x) }) {
        ptr::write(
            ptr,
            Default::default(),
        );
    }

    growable
}

/// Note, this does not drop the actual chunks.
unsafe fn drop_growable<T>(ptr: *mut Growable<T>) {
    assert_eq!(
        (*atomic_count_of(ptr)).load(Ordering::Relaxed),
        0,
        "Internal error, dropping a growable without having all instances of it detached!"
    );

    let total_len = total_len_of(ptr);
    let num_chunks_prior = total_len - chunks_of(ptr);
    let ptr = ptr.cast::<UCChunk<T>>().sub(num_chunks_prior);
    let layout = Layout::array::<UCChunk<T>>(total_len).unwrap();
    std::alloc::dealloc(ptr.cast(), layout);
}

/// Assumes the items in the boxed slices are already
/// dropped.
unsafe fn drop_recursive_growable<T>(ptr: *mut Growable<T>) {
    assert_eq!(
        (*atomic_count_of(ptr)).load(Ordering::Relaxed),
        0,
        "Internal error, dropping a growable without having all instances of it detached!"
    );
    let len = chunks_of(ptr);
    for ptr in (0..len).map(|x| ptr.cast::<UCChunk<T>>().add(x)) {
        ManuallyDrop::<Box<_>>::drop(&mut *std::ptr::addr_of_mut!(*(*ptr).get()));
    }
    drop_growable(ptr);
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
