use basis::{IndexSet, IndexT, Projection};
use core::*;
use geometry::{Matrix, Vector, norms::l1};
use std::{collections::HashMap, mem::replace};

#[derive(Clone, Serialize, Deserialize)]
pub struct VectorFunction {
    pub weights: Matrix<f64>,
}

impl VectorFunction {
    pub fn new(n_features: usize, n_outputs: usize) -> Self {
        VectorFunction {
            weights: Matrix::zeros((n_features, n_outputs)),
        }
    }

    fn append_weight_rows(&mut self, new_rows: Vec<Vec<f64>>) {
        let n_cols = self.weights.cols();
        let n_rows = self.weights.rows();
        let n_rows_new = new_rows.len();

        // Weight matrix stored in row-major format.
        let mut weights =
            unsafe { replace(&mut self.weights, Matrix::uninitialized((0, 0))).into_raw_vec() };

        weights.reserve_exact(n_rows_new);

        for row in new_rows {
            weights.extend(row);
        }

        self.weights = Matrix::from_shape_vec((n_rows + n_rows_new, n_cols), weights).unwrap();
    }
}

impl Approximator<Projection> for VectorFunction {
    type Value = Vector<f64>;

    fn evaluate(&self, p: &Projection) -> EvaluationResult<Vector<f64>> {
        Ok(match p {
            &Projection::Dense(ref dense) => self.weights.t().dot(dense),
            &Projection::Sparse(ref sparse) => (0..self.weights.cols())
                .map(|c| sparse
                    .iter()
                    .fold(0.0, |acc, idx| acc + self.weights[(*idx, c)])
                )
                .collect(),
        })
    }

    fn update(&mut self, p: &Projection, errors: Vector<f64>) -> UpdateResult<()> {
        Ok(match p {
            &Projection::Dense(ref dense) => {
                let scaled_errors = errors / l1(dense.as_slice().unwrap());
                let phi_matrix = dense.view().into_shape((dense.len(), 1)).unwrap();
                let error_matrix =
                    scaled_errors.view().into_shape((1, self.weights.cols())).unwrap();

                self.weights += &phi_matrix.dot(&error_matrix)
            }
            &Projection::Sparse(ref sparse) => for c in 0..self.weights.cols() {
                let mut col = self.weights.column_mut(c);
                let scaled_error = errors[c] / sparse.len() as f64;

                for idx in sparse {
                    col[*idx] += scaled_error
                }
            },
        })
    }

    fn adapt(&mut self, new_features: &HashMap<IndexT, IndexSet>) -> AdaptResult<usize> {
        let n_nfs = new_features.len();
        let n_outputs = self.weights.cols();

        let max_index = self.weights.len() + n_nfs - 1;

        let new_weights: Result<Vec<Vec<f64>>, _> = new_features
            .into_iter()
            .map(|(&i, idx)| {
                if i > max_index {
                    Err(AdaptError::Failed)
                } else {
                    Ok((0..n_outputs)
                        .map(|c| {
                            let c = self.weights.column(c);

                            idx.iter().fold(0.0, |acc, r| acc + c[*r])
                        })
                        .collect())
                }
            })
            .collect();

        match new_weights {
            Ok(new_weights) => {
                self.append_weight_rows(new_weights);

                Ok(n_nfs)
            }
            Err(err) => Err(err),
        }
    }
}

impl Parameterised for VectorFunction {
    fn weights(&self) -> Matrix<f64> {
        self.weights.clone()
    }
}

#[cfg(test)]
mod tests {
    extern crate seahash;

    use ::LFA;
    use approximators::VectorFunction;
    use basis::fixed::{Fourier, TileCoding};
    use core::Approximator;
    use geometry::Vector;
    use std::{collections::{BTreeSet, HashMap}, hash::BuildHasherDefault};

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let p = TileCoding::new(SHBuilder::default(), 4, 100);
        let mut f = LFA::vector_valued(p, 2);
        let input = vec![5.0];

        let _ = f.update(input.as_slice(), Vector::from_vec(vec![20.0, 50.0]));
        let out = f.evaluate(input.as_slice()).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let p = Fourier::new(3, vec![(0.0, 10.0)]);
        let mut f = LFA::vector_valued(p, 2);

        let input = vec![5.0];

        let _ = f.update(input.as_slice(), Vector::from_vec(vec![20.0, 50.0]));
        let out = f.evaluate(input.as_slice()).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_adapt() {
        let mut f = VectorFunction::new(100, 2);

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
            }
            Err(err) => panic!("VectorFunction::adapt failed with AdaptError::{:?}", err),
        }
    }
}
