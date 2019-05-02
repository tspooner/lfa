use crate::{
    basis::Projector,
    core::Features,
    geometry::{Card, Space},
};
use rand::{
    distributions::{self as dists, Distribution},
    thread_rng,
};

/// Fixed uniform basis projector.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
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
    pub fn uniform(n_features: usize, low: f64, high: f64) -> Self {
        Random::new(n_features, dists::Uniform::new_inclusive(low, high))
    }
}

impl<D: Distribution<f64>> Space for Random<D> {
    type Value = Features;

    fn dim(&self) -> usize { self.n_features }

    fn card(&self) -> Card { unimplemented!() }
}

impl<I: ?Sized, D: Distribution<f64>> Projector<I> for Random<D> {
    fn project(&self, _: &I) -> Features {
        let mut rng = thread_rng();

        (0..self.n_features)
            .into_iter()
            .map(|_| self.distribution.sample(&mut rng))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{quickcheck, TestResult};

    #[test]
    fn test_project_normal() {
        fn prop_output(length: usize, mean: f64, std: f64, input: Vec<f64>) -> TestResult {
            if std < 0.0 {
                TestResult::discard()
            } else {
                match Random::normal(length, mean, std).project(&input) {
                    Features::Sparse(_) => TestResult::failed(),
                    Features::Dense(activations) => {
                        TestResult::from_bool(activations.len() == length)
                    },
                }
            }
        }

        quickcheck(prop_output as fn(usize, f64, f64, Vec<f64>) -> TestResult);
    }

    #[test]
    fn test_project_lognormal() {
        fn prop_output(length: usize, mean: f64, std: f64, input: Vec<f64>) -> TestResult {
            if std < 0.0 {
                TestResult::discard()
            } else {
                match Random::log_normal(length, mean, std).project(&input) {
                    Features::Sparse(_) => TestResult::failed(),
                    Features::Dense(activations) => TestResult::from_bool(
                        activations.len() == length && activations.iter().all(|&v| v > 0.0),
                    ),
                }
            }
        }

        quickcheck(prop_output as fn(usize, f64, f64, Vec<f64>) -> TestResult);
    }

    #[test]
    fn test_project_gamma() {
        fn prop_output(length: usize, shape: f64, scale: f64, input: Vec<f64>) -> TestResult {
            if shape <= 0.0 || scale <= 0.0 {
                TestResult::discard()
            } else {
                match Random::gamma(length, shape, scale).project(&input) {
                    Features::Sparse(_) => TestResult::failed(),
                    Features::Dense(activations) => TestResult::from_bool(
                        activations.len() == length && activations.iter().all(|&v| v > 0.0),
                    ),
                }
            }
        }

        quickcheck(prop_output as fn(usize, f64, f64, Vec<f64>) -> TestResult);
    }

    #[test]
    fn test_project_uniform() {
        fn prop_output(length: usize, lb: f64, ub: f64, input: Vec<f64>) -> TestResult {
            if ub < lb {
                TestResult::discard()
            } else {
                match Random::uniform(length, lb, ub).project(&input) {
                    Features::Sparse(_) => TestResult::failed(),
                    Features::Dense(activations) => TestResult::from_bool(
                        activations.len() == length
                            && activations.into_iter().all(|&v| v >= lb && v < ub),
                    ),
                }
            }
        }

        quickcheck(prop_output as fn(usize, f64, f64, Vec<f64>) -> TestResult);
    }
}
