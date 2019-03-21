use crate::{
    core::*,
    geometry::{Matrix, Vector},
};
use std::mem::replace;

/// Weight-`Projection` evaluator with scalar `f64` output.
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

    fn extend_weights(&mut self, new_weights: Vec<f64>) {
        let mut weights =
            unsafe { replace(&mut self.weights, Vector::uninitialized((0,))).into_raw_vec() };

        weights.extend(new_weights);

        self.weights = Vector::from_vec(weights);
    }
}

impl Approximator<Projection> for ScalarFunction {
    type Output = f64;

    fn n_outputs(&self) -> usize { 1 }

    fn evaluate(&self, p: &Projection) -> EvaluationResult<f64> { Ok(p.dot(&self.weights)) }

    fn update(&mut self, p: &Projection, error: f64) -> UpdateResult<()> {
        Ok(match p {
            &Projection::Dense(ref activations) => self.weights.scaled_add(error, activations),
            &Projection::Sparse(ref indices) => {
                let scaled_error = error / indices.len() as f64;

                for idx in indices {
                    self.weights[*idx] += scaled_error;
                }
            },
        })
    }
}

impl Parameterised for ScalarFunction {
    fn weights(&self) -> Matrix<f64> {
        let n_rows = self.weights.len();

        self.weights.clone().into_shape((n_rows, 1)).unwrap()
    }

    fn n_weights(&self) -> usize { self.weights.len() }
}

#[cfg(test)]
mod tests {
    extern crate seahash;

    use crate::{
        core::Approximator,
        basis::{
            Composable,
            fixed::{Fourier, TileCoding},
        },
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
        let p = TileCoding::new(SHBuilder::default(), 4, 100);
        let mut f = LFA::scalar(p);
        let input = vec![5.0];

        let _ = f.update(&input, 50.0);
        let out = f.evaluate(&input).unwrap();

        assert!((out - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let p = Fourier::new(3, vec![(0.0, 10.0)]).normalise_l2();
        let mut f = LFA::scalar(p);

        let input = vec![5.0];

        let _ = f.update(input.as_slice(), 50.0);
        let out = f.evaluate(input.as_slice()).unwrap();

        println!("{}", out);

        assert!((out - 50.0).abs() < 1e-6);
    }
}
