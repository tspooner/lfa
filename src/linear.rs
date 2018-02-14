use super::{Approximator, EvaluationResult, UpdateResult, Projection, Projector};
use geometry::{Matrix, Vector};
use std::marker::PhantomData;

#[derive(Clone, Serialize, Deserialize)]
pub struct Linear<I: ?Sized, P: Projector<I>> {
    pub projector: P,
    pub weights: Matrix,

    phantom: PhantomData<I>,
}

impl<I: ?Sized, P: Projector<I>> Linear<I, P> {
    pub fn new(projector: P, n_outputs: usize) -> Self {
        let n_features = projector.size();

        Self {
            projector: projector,
            weights: Matrix::zeros((n_features, n_outputs)),

            phantom: PhantomData,
        }
    }

    pub fn assign(&mut self, values: Matrix<f64>) { self.weights = values; }

    pub fn assign_cols(&mut self, values: Vector<f64>) {
        self.weights = values.broadcast(self.weights.dim()).unwrap().to_owned();
    }

    pub fn evaluate_full(&self, p: &Projection) -> Vector<f64> {
        match p {
            &Projection::Dense(ref dense) => self.weights.t().dot(&(dense/p.z())),
            &Projection::Sparse(ref sparse) => (0..self.weights.cols()).map(|c| {
                sparse.iter().fold(0.0, |acc, idx| acc + self.weights[(*idx, c)])
            }).collect(),
        }
    }

    pub fn update_full(&mut self, p: &Projection, errors: Vector<f64>) {
        let z = p.z();
        let sf = 1.0/z;

        match p {
            &Projection::Dense(ref dense) => {
                let view = dense.view().into_shape((self.weights.rows(), 1)).unwrap();
                let error_matrix =
                    errors.view().into_shape((1, self.weights.cols())).unwrap();

                self.weights.scaled_add(sf, &view.dot(&error_matrix))
            },
            &Projection::Sparse(ref sparse) => {
                for c in 0..self.weights.cols() {
                    let mut col = self.weights.column_mut(c);
                    let error = errors[c];

                    for idx in sparse {
                        col[*idx] += error
                    }
                }
            },
        }
    }

    pub fn evaluate_column(&self, p: &Projection, col: usize) -> f64 {
        let col = self.weights.column(col);

        match p {
            &Projection::Dense(ref dense) => col.dot(&(dense/p.z())),
            &Projection::Sparse(ref sparse) =>
                sparse.iter().fold(0.0, |acc, idx| acc + col[*idx]),
        }
    }


    pub fn update_column(&mut self, p: &Projection, col: usize, error: f64) {
        let mut col = self.weights.column_mut(col);
        let scaled_error = error/p.z();

        match p {
            &Projection::Dense(ref dense) => col.scaled_add(scaled_error, dense),
            &Projection::Sparse(ref sparse) => {
                for idx in sparse {
                    col[*idx] += scaled_error
                }
            },
        }
    }
}

impl<I: ?Sized, P: Projector<I>> Approximator<I, f64> for Linear<I, P> {
    fn evaluate(&self, input: &I) -> EvaluationResult<f64> {
        let p = self.projector.project(input);

        Ok(self.evaluate_column(&p, 0))
    }

    fn update(&mut self, input: &I, error: f64) -> UpdateResult<()> {
        let p = self.projector.project(input);

        Ok(self.update_column(&p, 0, error))
    }
}

impl<I: ?Sized, P: Projector<I>> Approximator<I, Vector<f64>> for Linear<I, P> {
    fn evaluate(&self, input: &I) -> EvaluationResult<Vector<f64>> {
        let p = self.projector.project(input);

        Ok(self.evaluate_full(&p))
    }

    fn update(&mut self, input: &I, errors: Vector<f64>) -> UpdateResult<()> {
        let p = self.projector.project(input);

        Ok(self.update_full(&p, errors))
    }
}

#[cfg(test)]
mod tests {
    extern crate seahash;

    use super::*;
    use projection::TileCoding;
    use std::hash::BuildHasherDefault;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_dense_update_eval() {
        let p = TileCoding::new(SHBuilder::default(), 4, 100);
        let mut f = Linear::new(p.clone(), 1);

        let input = vec![5.0];

        f.update(input.as_slice(), 50.0);
        let out: f64 = f.evaluate(input.as_slice()).unwrap();

        assert!(out > 0.0);
    }
}
