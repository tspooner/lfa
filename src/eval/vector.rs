use crate::{
    core::*,
    geometry::{Matrix, MatrixView, MatrixViewMut, Vector},
};

/// Weight-`Projection` evaluator with vector `Vector<f64>` output.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Parameterised for VectorFunction {
    fn weights(&self) -> Matrix<f64> { self.weights.clone() }
    fn weights_view(&self) -> MatrixView<f64> { self.weights.view() }
    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> { self.weights.view_mut() }
}

impl Approximator for VectorFunction {
    type Output = Vector<f64>;

    fn n_outputs(&self) -> usize { self.weights.cols() }

    fn evaluate(&self, features: &Features) -> EvaluationResult<Self::Output> {
        apply_to_features!(features => activations, {
            Ok(activations.dot(&self.weights))
        }; indices, {
            Ok(self.weights.gencolumns().into_iter().map(|col| {
                Features::dot_sparse(indices, &col)
            }).collect())
        })
    }

    fn jacobian(&self, features: &Features) -> Matrix<f64> {
        let dim = self.weights_dim();
        let phi = features.expanded(dim.0);

        let mut g = unsafe { Matrix::uninitialized(dim) };

        g.gencolumns_mut().into_iter().for_each(|mut c| {
            c.assign(&phi);
        });

        g
    }

    fn update_grad(&mut self, grad: &Matrix<f64>, update: Self::Output) -> UpdateResult<()> {
        Ok(update.into_iter().enumerate().for_each(|(c, &e)| {
            self.weights.column_mut(c).scaled_add(e, &grad.column(c));
        }))
    }

    fn update(&mut self, features: &Features, errors: Self::Output) -> UpdateResult<()> {
        apply_to_features!(features => activations, {
            Ok(for (c, &e) in errors.into_iter().enumerate() {
                self.weights.column_mut(c).scaled_add(e, activations);
            })
        }; indices, {
            Ok(for (c, &e) in errors.into_iter().enumerate() {
                let scaled_error = e / indices.len() as f64;
                let mut col = self.weights.column_mut(c);

                for idx in indices {
                    col[*idx] += scaled_error
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
        geometry::{Space, Vector},
    };
    use std::hash::BuildHasherDefault;
    use super::VectorFunction;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let projector = TileCoding::new(SHBuilder::default(), 4, 100);
        let mut evaluator = VectorFunction::zeros(projector.dim(), 2);

        assert_eq!(evaluator.n_outputs(), 2);
        assert_eq!(evaluator.weights.len(), 200);

        let features = projector.project(&vec![5.0]);

        let _ = evaluator.update(&features, Vector::from_vec(vec![20.0, 50.0]));
        let out = evaluator.evaluate(&features).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let projector = Fourier::new(3, vec![(0.0, 10.0)]).normalise_l2();
        let mut evaluator = VectorFunction::zeros(projector.dim(), 2);

        assert_eq!(evaluator.n_outputs(), 2);
        assert_eq!(evaluator.weights.len(), 6);

        let features = projector.project(&vec![5.0]);

        let _ = evaluator.update(&features, Vector::from_vec(vec![20.0, 50.0]));
        let out = evaluator.evaluate(&features).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }
}
