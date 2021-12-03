use criterion::{black_box, criterion_group, criterion_main, Criterion};
use crossbeam::channel::unbounded;

#[path = "../common.rs"]
mod common;
use common::bench_impl;

fn crossbeam_acc_impl_mt(n: usize) {
    let (sender, receiver) = unbounded();
    let handles = (0..7).map(|_| {
        let sender = sender.clone();
        let receiver = receiver.clone();

        std::thread::spawn(move || {
            for x in 0..n {
                sender.send(black_box(x)).unwrap();
            }
            for _ in 0..n {
                black_box(receiver.recv().unwrap());
            }
        })
    })
        .collect::<Vec<_>>();

    handles.into_iter().for_each(|x| x.join().unwrap());
}

fn crossbeam_imm_impl_mt(n: usize) {
    let (sender, receiver) = unbounded();
    let handles = (0..7).map(|_| {
        let sender = sender.clone();
        let receiver = receiver.clone();

        std::thread::spawn(move || {
            for x in 0..n {
                sender.send(black_box(x)).unwrap();
                black_box(receiver.recv().unwrap());
            }
        })
    })
        .collect::<Vec<_>>();

    handles.into_iter().for_each(|x| x.join().unwrap());
}

fn crossbeam_acc_std_compat_mt(n: usize) {
    let (sender, receiver) = unbounded();
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

fn crossbeam_imm_std_compat_mt(n: usize) {
    let (sender, receiver) = unbounded();
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

fn crossbeam_acc(c: &mut Criterion) {
    bench_impl(crossbeam_acc_impl_mt)(c);
}

fn crossbeam_imm(c: &mut Criterion) {
    bench_impl(crossbeam_imm_impl_mt)(c);
}

fn crossbeam_acc_std_compat(c: &mut Criterion) {
    bench_impl(crossbeam_acc_std_compat_mt)(c);
}

fn crossbeam_imm_std_compat(c: &mut Criterion) {
    bench_impl(crossbeam_imm_std_compat_mt)(c);
}

criterion_group!(
    crossbeam_channels_mt,
    crossbeam_imm,
    crossbeam_acc,
    crossbeam_acc_std_compat,
    crossbeam_imm_std_compat,
);

criterion_main!(crossbeam_channels_mt);
