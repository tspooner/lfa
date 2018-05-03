use approximators::Approximator;
use error::AdaptError;
use geometry::{Vector, Matrix};
use projectors::{Projection, IndexT, IndexSet};
use std::collections::HashMap;
use std::mem::replace;
use {EvaluationResult, UpdateResult, AdaptResult};

#[derive(Clone, Serialize, Deserialize)]
pub struct Multi {
    pub weights: Matrix<f64>,
}

impl Multi {
    pub fn new(n_features: usize, n_outputs: usize) -> Self {
        Multi {
            weights: Matrix::zeros((n_features, n_outputs)),
        }
    }

    pub fn assign(&mut self, rhs: &Matrix<f64>) { self.weights.assign(rhs); }

    pub fn assign_cols(&mut self, rhs: &Vector<f64>) {
        let view = rhs.broadcast(self.weights.dim()).unwrap();

        self.weights.assign(&view);
    }

    fn append_weight_rows(&mut self, new_rows: Vec<Vec<f64>>) {
        let n_cols = self.weights.cols();
        let n_rows = self.weights.rows();
        let n_rows_new = new_rows.len();

        // Weight matrix stored in row-major format.
        let mut weights = unsafe {
            replace(&mut self.weights, Matrix::uninitialized((0, 0))).into_raw_vec()
        };

        weights.reserve_exact(n_rows_new);

        for row in new_rows {
            weights.extend(row);
        }

        self.weights = Matrix::from_shape_vec((n_rows+n_rows_new, n_cols), weights).unwrap();
    }
}

impl Approximator<Projection> for Multi {
    type Value = Vector<f64>;

    fn evaluate(&self, p: &Projection) -> EvaluationResult<Vector<f64>> {
        Ok(match p {
            &Projection::Dense(ref dense) => self.weights.t().dot(&(dense / p.z())),
            &Projection::Sparse(ref sparse) => (0..self.weights.cols())
                .map(|c| {
                    sparse
                        .iter()
                        .fold(0.0, |acc, idx| acc + self.weights[(*idx, c)])
                })
                .collect(),
        })
    }

    fn update(&mut self, p: &Projection, errors: Vector<f64>) -> UpdateResult<()> {
        let z = p.z();

        Ok(match p {
            &Projection::Dense(ref dense) => {
                let view = dense.view().into_shape((self.weights.rows(), 1)).unwrap();
                let error_matrix = errors.view().into_shape((1, self.weights.cols())).unwrap();

                self.weights.scaled_add(1.0 / z, &view.dot(&error_matrix))
            },
            &Projection::Sparse(ref sparse) => for c in 0..self.weights.cols() {
                let mut col = self.weights.column_mut(c);
                let error = errors[c];
                let scaled_error = error / z;

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

        let new_weights: Result<Vec<Vec<f64>>, _> = new_features.into_iter().map(|(&i, idx)| {
            if i > max_index {
                Err(AdaptError::Failed)
            } else {
                Ok((0..n_outputs).map(|c| {
                    let c = self.weights.column(c);

                    idx.iter().fold(0.0, |acc, r| acc + c[*r])
                }).collect())
            }
        }).collect();

        match new_weights {
            Ok(new_weights) => {
                self.append_weight_rows(new_weights);

                Ok(n_nfs)
            },
            Err(err) => Err(err)
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate seahash;

    use LFA;
    use approximators::Approximator;
    use geometry::Vector;
    use projectors::fixed::{Fourier, TileCoding};
    use std::hash::BuildHasherDefault;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let p = TileCoding::new(SHBuilder::default(), 4, 100);
        let mut f = LFA::multi(p, 2);

        let input = vec![5.0];

        let _ = f.update(input.as_slice(), Vector::from_vec(vec![20.0, 50.0]));
        let out = f.evaluate(input.as_slice()).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let p = Fourier::new(3, vec![(0.0, 10.0)]);
        let mut f = LFA::multi(p, 2);

        let input = vec![5.0];

        let _ = f.update(input.as_slice(), Vector::from_vec(vec![20.0, 50.0]));
        let out = f.evaluate(input.as_slice()).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }
}
