use crate::{
    basis::Projection,
    core::*,
    geometry::{norms::l1, Matrix, Vector},
};
use std::collections::HashMap;
use super::adapt_matrix;

/// Weight-`Projection` evaluator with vector `Vector<f64>` output.
#[derive(Clone, Serialize, Deserialize)]
pub struct VectorFunction {
    pub weights: Matrix<f64>,
}

impl VectorFunction {
    pub fn new(weights: Matrix<f64>) -> Self {
        VectorFunction { weights, }
    }

    pub fn zeros(n_features: usize, n_outputs: usize) -> Self {
        VectorFunction {
            weights: Matrix::zeros((n_features, n_outputs)),
        }
    }
}

impl Approximator<Projection> for VectorFunction {
    type Value = Vector<f64>;

    fn n_outputs(&self) -> usize { self.weights.cols() }

    fn evaluate(&self, p: &Projection) -> EvaluationResult<Vector<f64>> {
        Ok(p.matmul(&self.weights))
    }

    fn update(&mut self, p: &Projection, errors: Vector<f64>) -> UpdateResult<()> {
        Ok(match p {
            &Projection::Dense(ref activations) => {
                let scaled_errors = errors / l1(activations.as_slice().unwrap());
                let phi_matrix = activations
                    .view()
                    .into_shape((activations.len(), 1))
                    .unwrap();
                let error_matrix = scaled_errors
                    .view()
                    .into_shape((1, self.weights.cols()))
                    .unwrap();

                self.weights += &phi_matrix.dot(&error_matrix)
            },
            &Projection::Sparse(ref indices) => {
                for c in 0..self.weights.cols() {
                    let mut col = self.weights.column_mut(c);
                    let scaled_error = errors[c] / indices.len() as f64;

                    for idx in indices {
                        col[*idx] += scaled_error
                    }
                }
            },
        })
    }

    fn adapt(&mut self, new_features: &HashMap<IndexT, IndexSet>) -> AdaptResult<usize> {
        adapt_matrix(&mut self.weights, new_features)
    }
}

impl Parameterised for VectorFunction {
    fn weights(&self) -> Matrix<f64> { self.weights.clone() }

    fn n_weights(&self) -> usize { self.weights.len() }
}

#[cfg(test)]
mod tests {
    extern crate seahash;

    use crate::{
        core::Approximator,
        basis::fixed::{Fourier, TileCoding},
        geometry::Vector,
        LFA,
    };
    use std::{
        collections::{BTreeSet, HashMap},
        hash::BuildHasherDefault,
    };
    use super::VectorFunction;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let p = TileCoding::new(SHBuilder::default(), 4, 100);
        let mut f = LFA::vector(p, 2);
        let input = vec![5.0];

        let _ = f.update(input.as_slice(), Vector::from_vec(vec![20.0, 50.0]));
        let out = f.evaluate(input.as_slice()).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let p = Fourier::new(3, vec![(0.0, 10.0)]);
        let mut f = LFA::vector(p, 2);

        let input = vec![5.0];

        let _ = f.update(input.as_slice(), Vector::from_vec(vec![20.0, 50.0]));
        let out = f.evaluate(input.as_slice()).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_adapt() {
        let mut f = VectorFunction::zeros(100, 2);

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
            Err(err) => panic!("VectorFunction::adapt failed with AdaptError::{:?}", err),
        }
    }
}
