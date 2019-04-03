use crate::{
    core::*,
    geometry::{Matrix, MatrixView, MatrixViewMut},
};

/// Weight-`Projection` evaluator with pair `(f64, f64)` output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairFunction {
    pub weights: Matrix<f64>,
}

impl PairFunction {
    pub fn new(weights: Matrix<f64>) -> Self {
        PairFunction { weights, }
    }

    pub fn zeros(n_features: usize) -> Self {
        PairFunction::new(Matrix::zeros((n_features, 2)))
    }
}

impl Parameterised for PairFunction {
    fn weights(&self) -> Matrix<f64> { self.weights.clone() }
    fn weights_view(&self) -> MatrixView<f64> { self.weights.view() }
    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> { self.weights.view_mut() }
}

impl Approximator for PairFunction {
    type Output = (f64, f64);

    fn n_outputs(&self) -> usize { 2 }

    fn evaluate(&self, features: &Features) -> EvaluationResult<Self::Output> {
        apply_to_features!(features => activations, {
            Ok((
                self.weights.column(0).dot(activations),
                self.weights.column(1).dot(activations),
            ))
        }; indices, {
            Ok(indices.iter().fold((0.0, 0.0), |acc, idx| (
                acc.0 + self.weights[(*idx, 0)],
                acc.1 + self.weights[(*idx, 1)],
            )))
        })
    }

    fn update(&mut self, features: &Features, errors: Self::Output) -> UpdateResult<()> {
        apply_to_features!(features => activations, {
            Ok({
                self.weights.column_mut(0).scaled_add(errors.0, activations);
                self.weights.column_mut(1).scaled_add(errors.1, activations);
            })
        }; indices, {
            Ok({
                let z = indices.len() as f64;
                let scaled_errors = (errors.0 / z, errors.1 / z);

                indices.iter().for_each(|idx| {
                    self.weights[(*idx, 0)] += scaled_errors.0;
                    self.weights[(*idx, 1)] += scaled_errors.1;
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
    };
    use std::hash::BuildHasherDefault;
    use super::PairFunction;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let projector = TileCoding::new(SHBuilder::default(), 4, 100);
        let mut evaluator = PairFunction::zeros(projector.dim());

        let features = projector.project(&vec![5.0]);

        let _ = evaluator.update(&features, (20.0, 50.0));
        let out = evaluator.evaluate(&features).unwrap();

        assert!((out.0 - 20.0).abs() < 1e-6);
        assert!((out.1 - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let projector = Fourier::new(3, vec![(0.0, 10.0)]).normalise_l2();
        let mut evaluator = PairFunction::zeros(projector.dim());

        let features = projector.project(&vec![5.0]);

        let _ = evaluator.update(&features, (20.0, 50.0));
        let out = evaluator.evaluate(&features).unwrap();

        assert!((out.0 - 20.0).abs() < 1e-6);
        assert!((out.1 - 50.0).abs() < 1e-6);
    }
}
