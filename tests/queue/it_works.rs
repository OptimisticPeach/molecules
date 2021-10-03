use std::sync::Arc;
use criterion::black_box;
use molecules::queue::Queue;

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
fn multi_threaded() {
    let queue = Arc::new(Queue::<usize>::new());
    let handles = (0..14).map(|_| {
        let queue = queue.clone();

        std::thread::spawn(move || {
            // println!("Start! {:?}", std::thread::current().id());
            for _ in 0..240 {
                queue.push(black_box(b'W' as usize));
            }
            for _ in 0..240 {
                black_box(queue.pop().unwrap());
            }
            // println!("Done! {:?}", std::thread::current().id());
        })
    })
        .collect::<Vec<_>>();

    handles.into_iter().for_each(|x| x.join().unwrap());
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
