use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::sync::mpsc::channel;

#[path = "../common.rs"]
mod common;

use common::bench_impl;

fn std_acc_mt(n: usize) {
    let (sender, receiver) = channel();
    let handles = (0..7).map(|_| {
        let sender = sender.clone();

        std::thread::spawn(move || {
            for x in 0..n {
                sender.send(black_box(x)).unwrap();
            }
        })
    })
        .collect::<Vec<_>>();

    handles.into_iter().for_each(|x| x.join().unwrap());

    for _ in 0..n * 7 {
        black_box(receiver.try_recv().unwrap());
    }
}

fn std_imm_mt(n: usize) {
    let (sender, receiver) = channel();
    let handles = (0..7).map(|_| {
        let sender = sender.clone();

        std::thread::spawn(move || {
            for x in 0..n {
                sender.send(black_box(x)).unwrap();
            }
        })
    })
        .collect::<Vec<_>>();

    for _ in 0..n * 7 {
        black_box(receiver.recv().unwrap());
    }

    handles.into_iter().for_each(|x| x.join().unwrap());
}

fn std_acc(c: &mut Criterion) {
    bench_impl(std_acc_mt)(c);
}

fn std_imm(c: &mut Criterion) {
    bench_impl(std_imm_mt)(c);
}

criterion_group!(
    std_channels_mt,
    std_imm,
    std_acc,
);

criterion_main!(std_channels_mt);
