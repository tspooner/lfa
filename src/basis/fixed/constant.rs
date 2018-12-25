use crate::basis::{Projector, Projection};
use crate::geometry::{Card, Space};

/// Fixed uniform basis projector.
#[derive(Clone)]
pub struct Constant {
    n_features: usize,
    value: f64,
}

impl Constant {
    pub fn new(n_features: usize, value: f64) -> Self {
        Constant {
            n_features: n_features,
            value: value,
        }
    }

    pub fn zeros(n_features: usize) -> Self {
        Constant::new(n_features, 0.0)
    }

    pub fn ones(n_features: usize) -> Self {
        Constant::new(n_features, 1.0)
    }
}

impl Space for Constant {
    type Value = Projection;

    fn dim(&self) -> usize {
        self.n_features
    }

    fn card(&self) -> Card {
        unimplemented!()
    }
}

impl Projector<[f64]> for Constant {
    fn project(&self, _: &[f64]) -> Projection {
        vec![self.value; self.n_features].into()
    }
}

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;
    use super::*;

    #[test]
    fn test_project_zeros() {
        fn prop_output(length: usize, input: Vec<f64>) -> bool {
            match Constant::zeros(length).project(&input) {
                Projection::Sparse(_) => false,
                Projection::Dense(activations) => {
                    activations.len() == length && activations.into_iter().all(|&v| v == 0.0)
                },
            }
        }

        quickcheck(prop_output as fn(usize, Vec<f64>) -> bool);
    }

    #[test]
    fn test_project_ones() {
        fn prop_output(length: usize, input: Vec<f64>) -> bool {
            match Constant::ones(length).project(&input) {
                Projection::Sparse(_) => false,
                Projection::Dense(activations) => {
                    activations.len() == length && activations.into_iter().all(|&v| v == 1.0)
                },
            }
        }

        quickcheck(prop_output as fn(usize, Vec<f64>) -> bool);
    }

    #[test]
    fn test_project_general() {
        fn prop_output(length: usize, value: f64, input: Vec<f64>) -> bool {
            match Constant::new(length, value).project(&input) {
                Projection::Sparse(_) => false,
                Projection::Dense(activations) => {
                    activations.len() == length && activations.into_iter().all(|&v| v == value)
                },
            }
        }

        quickcheck(prop_output as fn(usize, f64, Vec<f64>) -> bool);
    }
}
