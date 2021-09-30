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
fn capacity_len_accurate() {
    let queue = Queue::<usize>::new();
    assert_eq!(queue.cap(), 1);
    assert_eq!(queue.len(), 0);

    queue.push(0);

    assert_eq!(queue.cap(), 1);
    assert_eq!(queue.len(), 1);

    queue.push(1);

    assert_eq!(queue.cap(), 5);
    assert_eq!(queue.len(), 2);

    queue.push(2);

    assert_eq!(queue.cap(), 5);
    assert_eq!(queue.len(), 3);

    assert_eq!(queue.pop(), Some(0));

    assert_eq!(queue.cap(), 5);
    assert_eq!(queue.len(), 2);

    assert_eq!(queue.pop(), Some(1));

    assert_eq!(queue.cap(), 5);
    assert_eq!(queue.len(), 1);

    assert_eq!(queue.pop(), Some(2));

    assert_eq!(queue.cap(), 5);
    assert_eq!(queue.len(), 0);
}
