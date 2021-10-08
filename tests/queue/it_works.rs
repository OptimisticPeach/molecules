use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use criterion::black_box;
use molecules::queue::{Chunk, Queue, UCChunk};

#[test]
fn it_works() {
    let queue = Queue::<usize>::new();
    queue.push(0);
    queue.push(1);
    queue.push(2);

    assert_eq!(queue.pop(), Some(0));
    assert_eq!(queue.pop(), Some(1));
    assert_eq!(queue.pop(), Some(2));
}

#[test]
fn pop_all() {
    for _ in 0..10000 {
        let queue = Arc::new(Queue::<usize>::new());

        let handles = (0..14).map(|_| {
            let queue = queue.clone();

            std::thread::spawn(move || {
                // println!("Start! {:?}", std::thread::current().id());
                for _ in 0..240 {
                    queue.push(black_box(b'W' as usize));
                }
                // println!("Done! {:?}", std::thread::current().id());
            })
        })
            .collect::<Vec<_>>();

        handles.into_iter().for_each(|x| x.join().unwrap());

        for _ in 0..240 * 14 {
            assert_eq!(black_box(queue.pop().unwrap()), b'W' as usize);
        }
    }
}

#[test]
fn push_all() {
    for _ in 0..10000 {
        let queue = Arc::new(Queue::<usize>::new());

        for _ in 0..240 * 14 {
            queue.push(black_box(b'W' as usize));
        }

        let handles = (0..14).map(|_| {
            let queue = queue.clone();

            std::thread::spawn(move || {
                // println!("Start! {:?}", std::thread::current().id());
                for _ in 0..240 {
                    assert_eq!(black_box(queue.pop().unwrap()), b'W' as usize);
                }
                // println!("Done! {:?}", std::thread::current().id());
            })
        })
            .collect::<Vec<_>>();

        handles.into_iter().for_each(|x| x.join().unwrap());
    }
}

#[test]
fn multi_threaded() {
    for _ in 0..1000 {
        let queue = Arc::new(Queue::<usize>::new());
        let handles = (0..7).map(|_| {
            let queue = queue.clone();

            std::thread::spawn(move || {
                // println!("Start! {:?}", std::thread::current().id());
                for x in 0..125 {
                    queue.push(black_box(x));
                }
                for _ in 0..125 {
                    black_box(queue.pop().unwrap());
                }
                // println!("Done! {:?}", std::thread::current().id());
            })
        })
            .collect::<Vec<_>>();

        handles.into_iter().for_each(|x| x.join().unwrap());
    }
}

#[test]
fn chunk_test() {
    for _ in 0..10000 {
        let queue = Arc::new(Chunk::<usize, 1024>::new());
        let reset_count = Arc::new(AtomicUsize::new(0));

        let handles = (0..16).map(|_| {
            let queue = queue.clone();
            let reset_count = reset_count.clone();

            std::thread::spawn(move || {
                // println!("Start! {:?}", std::thread::current().id());
                for _ in 0..64 {
                    assert!(!queue.is_full());
                    queue.push(black_box(b'W' as usize)).unwrap();
                }
                for _ in 0..64 {
                    assert!(!queue.is_finished());
                    assert_eq!(black_box(queue.pop().unwrap()), b'W' as usize);
                }
                if queue.try_reset().is_ok() {
                    reset_count.fetch_add(1, Ordering::SeqCst);
                }
                // println!("Done! {:?}", std::thread::current().id());
            })
        })
            .collect::<Vec<_>>();

        handles.into_iter().for_each(|x| x.join().unwrap());
        assert!(queue.is_new());
        assert_eq!(reset_count.load(Ordering::SeqCst), 1);
    }
}

// #[test]
// fn capacity_len_accurate() {
//     let queue = Queue::<usize>::new();
//     assert_eq!(queue.cap(), 1);
//     assert_eq!(queue.len(), 0);
//
//     queue.push(0);
//
//     assert_eq!(queue.cap(), 1);
//     assert_eq!(queue.len(), 1);
//
//     queue.push(1);
//
//     assert_eq!(queue.cap(), 5);
//     assert_eq!(queue.len(), 2);
//
//     queue.push(2);
//
//     assert_eq!(queue.cap(), 5);
//     assert_eq!(queue.len(), 3);
//
//     assert_eq!(queue.pop(), Some(0));
//
//     assert_eq!(queue.cap(), 5);
//     assert_eq!(queue.len(), 2);
//
//     assert_eq!(queue.pop(), Some(1));
//
//     assert_eq!(queue.cap(), 5);
//     assert_eq!(queue.len(), 1);
//
//     assert_eq!(queue.pop(), Some(2));
//
//     assert_eq!(queue.cap(), 5);
//     assert_eq!(queue.len(), 0);
// }
