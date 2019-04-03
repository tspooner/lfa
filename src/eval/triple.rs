use crate::{
    core::*,
    geometry::{Matrix, MatrixView, MatrixViewMut},
};

/// Weight-`Projection` evaluator with triple `(f64, f64, f64)` output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleFunction {
    pub weights: Matrix<f64>,
}

impl TripleFunction {
    pub fn new(weights: Matrix<f64>) -> Self {
        TripleFunction { weights, }
    }

    pub fn zeros(n_features: usize) -> Self {
        TripleFunction::new(Matrix::zeros((n_features, 3)))
    }
}

impl Parameterised for TripleFunction {
    fn weights(&self) -> Matrix<f64> { self.weights.clone() }
    fn weights_view(&self) -> MatrixView<f64> { self.weights.view() }
    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> { self.weights.view_mut() }
}

impl Approximator for TripleFunction {
    type Output = (f64, f64, f64);

    fn n_outputs(&self) -> usize { 3 }

    fn evaluate(&self, features: &Features) -> EvaluationResult<Self::Output> {
        apply_to_features!(features => activations, {
            Ok((
                self.weights.column(0).dot(activations),
                self.weights.column(1).dot(activations),
                self.weights.column(2).dot(activations),
            ))
        }; indices, {
            Ok(indices.iter().fold((0.0, 0.0, 0.0), |acc, idx| (
                acc.0 + self.weights[(*idx, 0)],
                acc.1 + self.weights[(*idx, 1)],
                acc.2 + self.weights[(*idx, 2)],
            )))
        })
    }

    fn update(&mut self, features: &Features, errors: Self::Output) -> UpdateResult<()> {
        apply_to_features!(features => activations, {
            Ok({
                self.weights.column_mut(0).scaled_add(errors.0, activations);
                self.weights.column_mut(1).scaled_add(errors.1, activations);
                self.weights.column_mut(2).scaled_add(errors.2, activations);
            })
        }; indices, {
            Ok({
                let z = indices.len() as f64;
                let scaled_errors = (errors.0 / z, errors.1 / z, errors.2 / z);

                indices.iter().for_each(|idx| {
                    self.weights[(*idx, 0)] += scaled_errors.0;
                    self.weights[(*idx, 1)] += scaled_errors.1;
                    self.weights[(*idx, 2)] += scaled_errors.2;
                });
            })
        })
    }
}

#[cfg(test)]
mod tests {
    extern crate seahash;

    use crate::{
        core::*,
        basis::{
            Composable,
            fixed::{Fourier, TileCoding},
        },
        geometry::Space,
        LFA,
    };
    use std::{
        collections::{BTreeSet, HashMap},
        hash::BuildHasherDefault,
    };
    use super::TripleFunction;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let projector = TileCoding::new(SHBuilder::default(), 4, 100);
        let mut evaluator = TripleFunction::zeros(projector.dim());

        let features = projector.project(&vec![5.0]);

        let _ = evaluator.update(&features, (20.0, 50.0, 100.0));
        let out = evaluator.evaluate(&features).unwrap();

        assert!((out.0 - 20.0).abs() < 1e-6);
        assert!((out.1 - 50.0).abs() < 1e-6);
        assert!((out.2 - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let projector = Fourier::new(3, vec![(0.0, 10.0)]).normalise_l2();
        let mut evaluator = TripleFunction::zeros(projector.dim());

        let features = projector.project(&vec![5.0]);

        let _ = evaluator.update(&features, (20.0, 50.0, 100.0));
        let out = evaluator.evaluate(&features).unwrap();

        assert!((out.0 - 20.0).abs() < 1e-6);
        assert!((out.1 - 50.0).abs() < 1e-6);
        assert!((out.2 - 100.0).abs() < 1e-6);
    }
}
