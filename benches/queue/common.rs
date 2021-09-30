use criterion::{black_box, BenchmarkId, Criterion};
use molecules::queue::Queue;

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

pub fn bench_impl<F: Fn(usize)>(f: F) -> impl FnOnce(&mut Criterion) {
    move |c| {
        let mut group = c.benchmark_group(std::any::type_name::<F>());
        for size in (1..6).map(|x| (5usize).pow(x)) {
            group.bench_function(BenchmarkId::from_parameter(size), |bencher| {
                bencher.iter(|| f(size))
            });
        }
        group.finish();
    }
}

pub fn molecules_acc(c: &mut Criterion) {
    bench_impl(molecules_acc_impl)(c);
}

pub fn molecules_imm(c: &mut Criterion) {
    bench_impl(molecules_imm_impl)(c);
}
