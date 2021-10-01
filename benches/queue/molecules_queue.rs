use criterion::{black_box, Criterion, criterion_main, criterion_group};
use molecules::queue::Queue;

#[path = "../common.rs"]
mod common;
use common::bench_impl;

fn molecules_acc_impl(n: usize) {
    let queue = Queue::<usize>::new();
    for x in 0..n {
        queue.push(black_box(x));
    }
    for x in 0..n {
        assert_eq!(queue.pop(), black_box(Some(x)));
    }
}

fn molecules_imm_impl(n: usize) {
    let queue = Queue::<usize>::new();
    for x in 0..n {
        queue.push(black_box(x));
        assert_eq!(queue.pop(), black_box(Some(x)));
    }
}

pub fn molecules_acc(c: &mut Criterion) {
    bench_impl(molecules_acc_impl)(c);
}

pub fn molecules_imm(c: &mut Criterion) {
    bench_impl(molecules_imm_impl)(c);
}

criterion_group!(
    molecules,
    molecules_imm,
    molecules_acc,
);
criterion_main!(molecules);
