use criterion::{black_box, criterion_group, criterion_main, Criterion};
use crossbeam::channel::unbounded;

#[path = "../common.rs"]
mod common;
use common::bench_impl;

fn crossbeam_acc_impl(n: usize) {
    let (sender, receiver) = unbounded();

    let mut sum = 0u64;
    for x in 0..n {
        sum += x as u64;
        sender.send(black_box(x)).unwrap();
    }
    for _ in 0..n {
        sum -= receiver.recv().unwrap() as u64;
    }

    assert_eq!(sum, 0);
}

fn crossbeam_imm_impl(n: usize) {
    let (sender, receiver) = unbounded();
    for x in 0..n {
        sender.send(black_box(x)).unwrap();
        assert_eq!(receiver.recv().unwrap(), black_box(x));
    }
}

fn crossbeam_acc(c: &mut Criterion) {
    bench_impl(crossbeam_acc_impl)(c);
}

fn crossbeam_imm(c: &mut Criterion) {
    bench_impl(crossbeam_imm_impl)(c);
}

criterion_group!(
    crossbeam_channels,
    crossbeam_imm,
    crossbeam_acc
);
criterion_main!(crossbeam_channels);
