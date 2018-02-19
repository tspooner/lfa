use {Approximator, EvaluationResult, UpdateResult, Projection, Projector};
use geometry::Vector;
use std::marker::PhantomData;

#[derive(Clone, Serialize, Deserialize)]
pub struct SimpleLinear<I: ?Sized, P: Projector<I>> {
    pub projector: P,
    pub weights: Vector<f64>,

    phantom: PhantomData<I>,
}

impl<I: ?Sized, P: Projector<I>> SimpleLinear<I, P> {
    pub fn new(projector: P) -> Self {
        let n_features = projector.size();

        Self {
            projector: projector,
            weights: Vector::zeros((n_features,)),

            phantom: PhantomData,
        }
    }

    pub fn assign(&mut self, values: &Vector<f64>) { self.weights.assign(values); }

    pub fn evaluate_projection(&self, p: &Projection) -> f64 {
        match p {
            &Projection::Dense(ref dense) => self.weights.dot(&(dense/p.z())),
            &Projection::Sparse(ref sparse) =>
                sparse.iter().fold(0.0, |acc, idx| acc + self.weights[*idx]),
        }
    }


    pub fn update_projection(&mut self, p: &Projection, error: f64) {
        let scaled_error = error/p.z();

        match p {
            &Projection::Dense(ref dense) => self.weights.scaled_add(scaled_error, dense),
            &Projection::Sparse(ref sparse) => {
                for idx in sparse {
                    self.weights[*idx] += scaled_error
                }
            },
        }
    }
}

impl<I: ?Sized, P: Projector<I>> Approximator<I> for SimpleLinear<I, P> {
    type Value = f64;

    fn evaluate(&self, input: &I) -> EvaluationResult<f64> {
        let p = self.projector.project(input);

        Ok(self.evaluate_projection(&p))
    }

    fn update(&mut self, input: &I, error: f64) -> UpdateResult<()> {
        let p = self.projector.project(input);

        Ok(self.update_projection(&p, error))
    }
}

#[cfg(test)]
mod tests {
    extern crate seahash;

    use super::*;
    use projection::{TileCoding, Fourier};
    use std::hash::BuildHasherDefault;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let p = TileCoding::new(SHBuilder::default(), 4, 100);
        let mut f = SimpleLinear::new(p.clone());

        let input = vec![5.0];

        let _ = f.update(input.as_slice(), 50.0);
        let out = f.evaluate(input.as_slice()).unwrap();

        assert!((out - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let p = Fourier::new(3, vec![(0.0, 10.0)]);
        let mut f = SimpleLinear::new(p.clone());

        let input = vec![5.0];

        let _ = f.update(input.as_slice(), 50.0);
        let out = f.evaluate(input.as_slice()).unwrap();

        assert!((out - 50.0).abs() < 1e-6);
    }
}
