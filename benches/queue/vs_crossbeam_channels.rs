use criterion::{black_box, criterion_group, criterion_main, Criterion};
use crossbeam::channel::unbounded;
use molecules::queue::Queue;

#[path = "common.rs"]
mod common;

use common::{bench_impl, molecules_acc, molecules_imm};

fn crossbeam_acc_impl(n: usize) {
    let (sender, receiver) = unbounded();
    for x in 0..n {
        sender.send(black_box(x)).unwrap();
    }
    for x in 0..n {
        assert_eq!(receiver.recv().unwrap(), black_box(x));
    }
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
    vs_crossbeam_channels,
    molecules_imm,
    molecules_acc,
    crossbeam_imm,
    crossbeam_acc
);
criterion_main!(vs_crossbeam_channels);
