use crate::basis::Projection;
use crate::core::*;
use crate::geometry::{norms::l1, Matrix};
use std::collections::HashMap;
use super::adapt_matrix;

/// Weight-`Projection` evaluator with pair `(f64, f64)` output.
#[derive(Clone, Serialize, Deserialize)]
pub struct PairFunction {
    pub weights: Matrix<f64>,
}

impl PairFunction {
    pub fn new(n_features: usize) -> Self {
        PairFunction {
            weights: Matrix::zeros((n_features, 2)),
        }
    }
}

impl Approximator<Projection> for PairFunction {
    type Value = (f64, f64);

    fn n_outputs(&self) -> usize { 2 }

    fn evaluate(&self, p: &Projection) -> EvaluationResult<(f64, f64)> {
        Ok(match p {
            &Projection::Dense(ref activations) => (
                self.weights.column(0).dot(activations),
                self.weights.column(1).dot(activations),
            ),
            &Projection::Sparse(ref indices) => indices.iter().fold((0.0, 0.0), |acc, idx| {
                (
                    acc.0 + self.weights[(*idx, 0)],
                    acc.1 + self.weights[(*idx, 1)],
                )
            }),
        })
    }

    fn update(&mut self, p: &Projection, errors: (f64, f64)) -> UpdateResult<()> {
        Ok(match p {
            &Projection::Dense(ref activations) => {
                let z = l1(activations.as_slice().unwrap());

                let phi_matrix = activations
                    .view()
                    .into_shape((activations.len(), 1))
                    .unwrap();
                let error_matrix =
                    Matrix::from_shape_vec((1, 2), vec![errors.0 / z, errors.1 / z]).unwrap();

                self.weights += &phi_matrix.dot(&error_matrix)
            },
            &Projection::Sparse(ref indices) => {
                let z = indices.len() as f64;
                let scaled_errors = (errors.0 / z, errors.1 / z);

                indices.iter().for_each(|idx| {
                    self.weights[(*idx, 0)] += scaled_errors.0;
                    self.weights[(*idx, 1)] += scaled_errors.1;
                });
            },
        })
    }

    fn adapt(&mut self, new_features: &HashMap<IndexT, IndexSet>) -> AdaptResult<usize> {
        adapt_matrix(&mut self.weights, new_features)
    }
}

impl Parameterised for PairFunction {
    fn weights(&self) -> Matrix<f64> { self.weights.clone() }
}

#[cfg(test)]
mod tests {
    extern crate seahash;

    use crate::{
        core::Approximator,
        basis::fixed::{Fourier, TileCoding},
        LFA,
    };
    use std::{
        collections::{BTreeSet, HashMap},
        hash::BuildHasherDefault,
    };
    use super::PairFunction;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let p = TileCoding::new(SHBuilder::default(), 4, 100);
        let mut f = LFA::pair_output(p);
        let input = vec![5.0];

        let _ = f.update(input.as_slice(), (20.0, 50.0));
        let out = f.evaluate(input.as_slice()).unwrap();

        assert!((out.0 - 20.0).abs() < 1e-6);
        assert!((out.1 - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let p = Fourier::new(3, vec![(0.0, 10.0)]);
        let mut f = LFA::pair_output(p);

        let input = vec![5.0];

        let _ = f.update(input.as_slice(), (20.0, 50.0));
        let out = f.evaluate(input.as_slice()).unwrap();

        assert!((out.0 - 20.0).abs() < 1e-6);
        assert!((out.1 - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_adapt() {
        let mut f = PairFunction::new(100);

        let mut new_features = HashMap::new();
        new_features.insert(100, {
            let mut idx = BTreeSet::new();

            idx.insert(10);
            idx.insert(90);

            idx
        });

        match f.adapt(&new_features) {
            Ok(n) => {
                assert_eq!(n, 1);
                assert_eq!(f.weights.rows(), 101);

                let c0 = f.weights.column(0);
                let c1 = f.weights.column(1);

                assert_eq!(c0[100], c0[10] / 2.0 + c0[90] / 2.0);
                assert_eq!(c1[100], c1[10] / 2.0 + c1[90] / 2.0);
            },
            Err(err) => panic!("PairFunction::adapt failed with AdaptError::{:?}", err),
        }
    }
}
