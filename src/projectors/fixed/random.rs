use geometry::{Card, Space};
use projectors::{Projection, Projector};
use rand::{Rng, thread_rng, rngs::ThreadRng, distributions::{self as dists, Distribution}};

/// Fixed uniform basis projector.
#[derive(Clone)]
pub struct Random<D: Distribution<f64>> {
    n_features: usize,
    distribution: D,
}

impl<D: Distribution<f64>> Random<D> {
    pub fn new(n_features: usize, distribution: D) -> Self {
        Random {
            n_features: n_features,
            distribution: distribution,
        }
    }
}

impl Random<dists::Normal> {
    pub fn normal(n_features: usize, mean: f64, std_dev: f64) -> Self {
        Random::new(n_features, dists::Normal::new(mean, std_dev))
    }
}

impl Random<dists::LogNormal> {
    pub fn log_normal(n_features: usize, mean: f64, std_dev: f64) -> Self {
        Random::new(n_features, dists::LogNormal::new(mean, std_dev))
    }
}

impl Random<dists::Gamma> {
    pub fn gamma(n_features: usize, shape: f64, scale: f64) -> Self {
        Random::new(n_features, dists::Gamma::new(shape, scale))
    }
}

impl Random<dists::Uniform<f64>> {
    pub fn range(n_features: usize, low: f64, high: f64) -> Self {
        Random::new(n_features, dists::Uniform::new_inclusive(low, high))
    }
}

impl<D: Distribution<f64>> Space for Random<D> {
    type Value = Projection;

    fn dim(&self) -> usize {
        self.n_features
    }

    fn card(&self) -> Card {
        unimplemented!()
    }
}

impl<D: Distribution<f64>> Projector<[f64]> for Random<D> {
    fn project(&self, _: &[f64]) -> Projection {
        let mut rng = thread_rng();

        (0..self.n_features).into_iter().map(|_| self.distribution.sample(&mut rng)).collect()
    }
}
