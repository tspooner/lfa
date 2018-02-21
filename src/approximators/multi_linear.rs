use {Approximator, EvaluationResult, Projection, Projector, UpdateResult};
use geometry::{Matrix, Vector};
use std::marker::PhantomData;

#[derive(Clone, Serialize, Deserialize)]
pub struct MultiLinear<I: ?Sized, P: Projector<I>> {
    pub projector: P,
    pub weights: Matrix<f64>,

    phantom: PhantomData<I>,
}

impl<I: ?Sized, P: Projector<I>> MultiLinear<I, P> {
    pub fn new(projector: P, n_outputs: usize) -> Self {
        let n_features = projector.span().into();

        Self {
            projector: projector,
            weights: Matrix::zeros((n_features, n_outputs)),

            phantom: PhantomData,
        }
    }

    pub fn assign(&mut self, values: &Matrix<f64>) { self.weights.assign(values); }

    pub fn assign_cols(&mut self, values: &Vector<f64>) {
        let view = values.broadcast(self.weights.dim()).unwrap();

        self.weights.assign(&view);
    }

    pub fn evaluate_projection(&self, p: &Projection) -> Vector<f64> {
        match p {
            &Projection::Dense(ref dense) => self.weights.t().dot(&(dense / p.z())),
            &Projection::Sparse(ref sparse) => (0..self.weights.cols())
                .map(|c| {
                    sparse
                        .iter()
                        .fold(0.0, |acc, idx| acc + self.weights[(*idx, c)])
                })
                .collect(),
        }
    }

    pub fn update_projection(&mut self, p: &Projection, errors: Vector<f64>) {
        let z = p.z();

        match p {
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
        }
    }
}

impl<I: ?Sized, P: Projector<I>> Approximator<I> for MultiLinear<I, P> {
    type Value = Vector<f64>;

    fn evaluate(&self, input: &I) -> EvaluationResult<Vector<f64>> {
        let p = self.projector.project(input);

        Ok(self.evaluate_projection(&p))
    }

    fn update(&mut self, input: &I, errors: Vector<f64>) -> UpdateResult<()> {
        let p = self.projector.project(input);

        Ok(self.update_projection(&p, errors))
    }
}

#[cfg(test)]
mod tests {
    extern crate seahash;

    use super::*;
    use projection::{Fourier, TileCoding};
    use std::hash::BuildHasherDefault;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let p = TileCoding::new(SHBuilder::default(), 4, 100);
        let mut f = MultiLinear::new(p.clone(), 2);

        let input = vec![5.0];

        let _ = f.update(input.as_slice(), Vector::from_vec(vec![20.0, 50.0]));
        let out = f.evaluate(input.as_slice()).unwrap();

        println!("{:?}", out);

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let p = Fourier::new(3, vec![(0.0, 10.0)]);
        let mut f = MultiLinear::new(p.clone(), 2);

        let input = vec![5.0];

        let _ = f.update(input.as_slice(), Vector::from_vec(vec![20.0, 50.0]));
        let out = f.evaluate(input.as_slice()).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }
}
