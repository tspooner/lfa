use crate::{
    core::*,
    geometry::{MatrixView, MatrixViewMut, Vector},
};
// use std::mem::replace;

/// Weight-`Features` evaluator with scalar `f64` output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarFunction {
    pub weights: Vector<f64>,
}

impl ScalarFunction {
    pub fn new(weights: Vector<f64>) -> Self {
        ScalarFunction { weights, }
    }

    pub fn zeros(n_features: usize) -> Self {
        ScalarFunction::new(Vector::zeros((n_features,)))
    }

    // fn extend_weights(&mut self, new_weights: Vec<f64>) {
        // let mut weights =
            // unsafe { replace(&mut self.weights, Vector::uninitialized((0,))).into_raw_vec() };

        // weights.extend(new_weights);

        // self.weights = Vector::from_vec(weights);
    // }
}

impl Parameterised for ScalarFunction {
    fn weights_view(&self) -> MatrixView<f64> {
        let n_rows = self.weights.len();

        self.weights.view().into_shape((n_rows, 1)).unwrap()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        let n_rows = self.weights.len();

        self.weights.view_mut().into_shape((n_rows, 1)).unwrap()
    }
}

impl Approximator for ScalarFunction {
    type Output = f64;

    fn n_outputs(&self) -> usize { 1 }

    fn evaluate(&self, features: &Features) -> EvaluationResult<Self::Output> {
        apply_to_projection!(features => activations, {
            Ok(activations.dot(&self.weights))
        }; indices, {
            Ok(Features::dot_sparse(indices, &self.weights.view()))
        })
    }

    fn update(&mut self, features: &Features, error: Self::Output) -> UpdateResult<()> {
        apply_to_projection!(features => activations, {
            Ok(self.weights.scaled_add(error, activations))
        }; indices, {
            Ok({
                let scaled_error = error / indices.len() as f64;

                for idx in indices {
                    self.weights[*idx] += scaled_error;
                }
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
    use super::ScalarFunction;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let projector = TileCoding::new(SHBuilder::default(), 4, 100);
        let mut evaluator = ScalarFunction::zeros(projector.dim());

        let features = projector.project(&vec![5.0]);

        let _ = evaluator.update(&features, 50.0);
        let out = evaluator.evaluate(&features).unwrap();

        assert!((out - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let projector = Fourier::new(3, vec![(0.0, 10.0)]).normalise_l2();
        let mut evaluator = ScalarFunction::zeros(projector.dim());

        let features = projector.project(&vec![5.0]);

        let _ = evaluator.update(&features, 50.0);
        let out = evaluator.evaluate(&features).unwrap();

        assert!((out - 50.0).abs() < 1e-6);
    }
}
