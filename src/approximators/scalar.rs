use basis::{IndexSet, IndexT, Projection};
use core::*;
use geometry::{Matrix, Vector, norms::l1};
use std::{collections::HashMap, mem::replace};

#[derive(Clone, Serialize, Deserialize)]
pub struct ScalarFunction {
    pub weights: Vector<f64>,
}

impl ScalarFunction {
    pub fn new(n_features: usize) -> Self {
        ScalarFunction {
            weights: Vector::zeros((n_features,)),
        }
    }

    fn extend_weights(&mut self, new_weights: Vec<f64>) {
        let mut weights =
            unsafe { replace(&mut self.weights, Vector::uninitialized((0,))).into_raw_vec() };

        weights.extend(new_weights);

        self.weights = Vector::from_vec(weights);
    }
}

impl Approximator<Projection> for ScalarFunction {
    type Value = f64;

    fn evaluate(&self, p: &Projection) -> EvaluationResult<f64> {
        Ok(match p {
            &Projection::Dense(ref dense) => self.weights.dot(dense),
            &Projection::Sparse(ref sparse) => {
                sparse.iter().fold(0.0, |acc, idx| acc + self.weights[*idx])
            }
        })
    }

    fn update(&mut self, p: &Projection, error: f64) -> UpdateResult<()> {
        Ok(match p {
            &Projection::Dense(ref dense) => {
                let scaled_error = error / l1(dense.as_slice().unwrap());

                self.weights.scaled_add(scaled_error, dense)
            },
            &Projection::Sparse(ref sparse) => {
                let scaled_error = error / sparse.len() as f64;

                for idx in sparse {
                    self.weights[*idx] += scaled_error;
                }
            },
        })
    }

    fn adapt(&mut self, new_features: &HashMap<IndexT, IndexSet>) -> AdaptResult<usize> {
        let n_nfs = new_features.len();
        let max_index = self.weights.len() + n_nfs - 1;

        let new_weights: Result<Vec<f64>, _> = new_features
            .into_iter()
            .map(|(&i, idx)| {
                if i > max_index {
                    Err(AdaptError::Failed)
                } else {
                    Ok(idx.iter().fold(0.0, |acc, j| acc + self.weights[*j]) / (idx.len() as f64))
                }
            })
            .collect();

        self.extend_weights(new_weights?);

        Ok(n_nfs)
    }
}

impl Parameterised for ScalarFunction {
    fn weights(&self) -> Matrix<f64> {
        let n_rows = self.weights.len();

        self.weights.clone().into_shape((n_rows, 1)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    extern crate seahash;

    use ::LFA;
    use approximators::ScalarFunction;
    use basis::fixed::{Fourier, TileCoding};
    use core::Approximator;
    use std::{collections::{BTreeSet, HashMap}, hash::BuildHasherDefault};

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let p = TileCoding::new(SHBuilder::default(), 4, 100);
        let mut f = LFA::scalar_valued(p);
        let input = vec![5.0];

        let _ = f.update(&input, 50.0);
        let out = f.evaluate(&input).unwrap();

        assert!((out - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let p = Fourier::new(3, vec![(0.0, 10.0)]);
        let mut f = LFA::scalar_valued(p);

        let input = vec![5.0];

        let _ = f.update(input.as_slice(), 50.0);
        let out = f.evaluate(input.as_slice()).unwrap();

        assert!((out - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_adapt() {
        let mut f = ScalarFunction::new(100);

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
                assert_eq!(f.weights.len(), 101);
                assert_eq!(f.weights[100], f.weights[10] / 2.0 + f.weights[90] / 2.0);
            }
            Err(err) => panic!("ScalarFunction::adapt failed with AdaptError::{:?}", err),
        }
    }
}
