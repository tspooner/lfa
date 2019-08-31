extern crate rand;
extern crate rand_distr;

use crate::{Result, Features, basis::Projector};
use self::{
    rand::{thread_rng, Rng},
    rand_distr::{self as dists, Distribution},
};

/// Fixed uniform basis projector.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
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

impl Random<dists::Normal<f64>> {
    pub fn normal(n_features: usize, mean: f64, std_dev: f64) -> Self {
        Random::new(n_features, dists::Normal::new(mean, std_dev).unwrap())
    }
}

impl Random<dists::LogNormal<f64>> {
    pub fn log_normal(n_features: usize, mean: f64, std_dev: f64) -> Self {
        Random::new(n_features, dists::LogNormal::new(mean, std_dev).unwrap())
    }
}

impl Random<dists::Gamma<f64>> {
    pub fn gamma(n_features: usize, shape: f64, scale: f64) -> Self {
        Random::new(n_features, dists::Gamma::new(shape, scale).unwrap())
    }
}

impl Random<dists::Uniform<f64>> {
    pub fn uniform(n_features: usize, low: f64, high: f64) -> Self {
        Random::new(n_features, dists::Uniform::new_inclusive(low, high))
    }
}

impl<D: Distribution<f64>> Projector for Random<D> {
    fn n_features(&self) -> usize { self.n_features }

    fn project_ith(&self, _: &[f64], _: usize) -> Result<Option<f64>> {
        Ok(Some(thread_rng().sample(&self.distribution)))
    }

    fn project(&self, _: &[f64]) -> Result<Features> {
        Ok(thread_rng().sample_iter(&self.distribution)
            .take(self.n_features)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::TestResult;

    #[test]
    fn test_project_normal() {
        quickcheck! {
            fn prop_output(length: usize, mean: f64, std: f64, input: Vec<f64>) -> TestResult {
                if std < 0.0 {
                    TestResult::discard()
                } else {
                    let features = Random::normal(length, mean, std).project(&input).unwrap();

                    TestResult::from_bool(features.n_features() == length)
                }
            }
        }
    }

    #[test]
    fn test_project_lognormal() {
        quickcheck! {
            fn prop_output(length: usize, mean: f64, std: f64, input: Vec<f64>) -> TestResult {
                if std < 0.0 {
                    TestResult::discard()
                } else {
                    let features = Random::log_normal(length, mean, std).project(&input).unwrap();

                    TestResult::from_bool(
                        features.n_features() == length &&
                        features.expanded().into_iter().all(|&v| v > 0.0)
                    )
                }
            }
        }
    }

    #[test]
    fn test_project_gamma() {
        quickcheck! {
            fn prop_output(length: usize, shape: f64, scale: f64, input: Vec<f64>) -> TestResult {
                if shape <= 0.0 || scale <= 0.0 {
                    TestResult::discard()
                } else {
                    let features = Random::gamma(length, shape, scale).project(&input).unwrap();

                    TestResult::from_bool(
                        features.n_features() == length &&
                        features.expanded().into_iter().all(|&v| v > 0.0)
                    )
                }
            }
        }
    }

    #[test]
    fn test_project_uniform() {
        quickcheck! {
            fn prop_output(length: usize, lb: f64, ub: f64, input: Vec<f64>) -> TestResult {
                if ub < lb {
                    TestResult::discard()
                } else {
                    let features = Random::uniform(length, lb, ub).project(&input).unwrap();

                    TestResult::from_bool(
                        features.n_features() == length &&
                        features.expanded().into_iter().all(|&v| v >= lb && v < ub)
                    )
                }
            }
        }
    }
}
