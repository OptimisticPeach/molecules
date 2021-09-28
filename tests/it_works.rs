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
