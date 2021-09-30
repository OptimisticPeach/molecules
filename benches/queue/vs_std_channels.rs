use std::sync::mpsc::channel;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use molecules::queue::Queue;

#[path = "common.rs"]
mod common;
use common::{bench_impl, molecules_acc, molecules_imm};

fn std_acc_impl(n: usize) {
    let (sender, receiver) = channel();
    for x in 0..n {
        sender.send(black_box(x)).unwrap();
    }
    for x in 0..n {
        assert_eq!(receiver.recv().unwrap(), black_box(x));
    }
}

fn std_imm_impl(n: usize) {
    let (sender, receiver) = channel();
    for x in 0..n {
        sender.send(black_box(x)).unwrap();
        assert_eq!(receiver.recv().unwrap(), black_box(x));
    }
}

fn std_acc(c: &mut Criterion) {
    bench_impl(std_acc_impl)(c);
}

fn std_imm(c: &mut Criterion) {
    bench_impl(std_imm_impl)(c);
}

criterion_group!(vs_std_channels, molecules_imm, molecules_acc, std_imm, std_acc);
criterion_main!(vs_std_channels);
