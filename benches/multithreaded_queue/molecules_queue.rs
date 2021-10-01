use std::sync::Arc;
use criterion::{black_box, Criterion, criterion_main, criterion_group};
use molecules::queue::Queue;

#[path = "../common.rs"]
mod common;
use common::bench_impl;

fn molecules_acc_impl_mt(n: usize) {
    let queue = Arc::new(Queue::<usize>::new());
    let handles = (0..7).map(|_| {
        let queue = queue.clone();

        std::thread::spawn(move || {
            for x in 0..n {
                queue.push(black_box(x));
            }
            for _ in 0..n {
                black_box(queue.pop().unwrap());
            }
        })
    })
        .collect::<Vec<_>>();

    handles.into_iter().for_each(|x| x.join().unwrap());
}

fn molecules_imm_impl_mt(n: usize) {
    let queue = Arc::new(Queue::<usize>::new());
    let handles = (0..7).map(|_| {
        let queue = queue.clone();

        std::thread::spawn(move || {
            for x in 0..n {
                queue.push(black_box(x));
                black_box(queue.pop().unwrap());
            }
        })
    })
        .collect::<Vec<_>>();

    handles.into_iter().for_each(|x| x.join().unwrap());
}

pub fn molecules_acc(c: &mut Criterion) {
    bench_impl(molecules_acc_impl_mt)(c);
}

pub fn molecules_imm(c: &mut Criterion) {
    bench_impl(molecules_imm_impl_mt)(c);
}

criterion_group!(
    molecules_mt,
    molecules_imm,
    molecules_acc,
);
criterion_main!(molecules_mt);