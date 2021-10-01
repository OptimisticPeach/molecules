use criterion::{BenchmarkId, Criterion};

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
