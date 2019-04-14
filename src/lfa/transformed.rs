use crate::{
    basis::Projector,
    core::*,
    eval::*,
    geometry::{Matrix, MatrixView, MatrixViewMut, Space},
    transforms::{Transform, Identity},
};
use elementwise::arithmetic::ElementwiseMul;

macro_rules! impl_builder {
    ($ftype:ty => $fname:ident) => {
        impl<P: Space, T> TransformedLFA<P, $ftype, T> {
            pub fn $fname(projector: P, transform: T) -> Self {
                let evaluator = <$ftype>::zeros(projector.dim());

                Self::new(projector, evaluator, transform)
            }
        }
    };
}

/// Transformed linear function approximator.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct TransformedLFA<P, E, T = Identity> {
    pub projector: P,
    pub evaluator: E,
    pub transform: T,
}

impl<P, E, T> TransformedLFA<P, E, T> {
    pub fn new(projector: P, evaluator: E, transform: T) -> Self {
        TransformedLFA { projector, evaluator, transform, }
    }
}

impl_builder!(ScalarFunction => scalar);
impl_builder!(PairFunction => pair);
impl_builder!(TripleFunction => triple);

impl<P: Space, T> TransformedLFA<P, VectorFunction, T> {
    pub fn vector(projector: P, n_outputs: usize, transform: T) -> Self {
        let evaluator = VectorFunction::zeros(projector.dim(), n_outputs);

        Self::new(projector, evaluator, transform)
    }
}

impl<P, E: Parameterised, T> Parameterised for TransformedLFA<P, E, T> {
    fn weights(&self) -> Matrix<f64> { self.evaluator.weights() }

    fn weights_view(&self) -> MatrixView<f64> { self.evaluator.weights_view() }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> { self.evaluator.weights_view_mut() }
}

impl<I: ?Sized, P, E, T> Embedding<I> for TransformedLFA<P, E, T>
where
    P: Projector<I>,
    E: Approximator,
{
    fn n_features(&self) -> usize {
        self.projector.dim()
    }

    fn embed(&self, input: &I) -> Features {
        self.projector.project(input)
    }
}

impl<P, E, T> Approximator for TransformedLFA<P, E, T>
where
    E: Approximator,
    T: Transform<E::Output, Output = E::Output>,
    T::Output: ElementwiseMul<T::Output> + IntoVector,
{
    type Output = T::Output;

    fn n_outputs(&self) -> usize {
        self.evaluator.n_outputs()
    }

    fn evaluate(&self, features: &Features) -> EvaluationResult<Self::Output> {
        self.evaluator.evaluate(features).map(|v| self.transform.transform(v))
    }

    fn jacobian(&self, features: &Features) -> Matrix<f64> {
        let v = self.evaluator.evaluate(features).unwrap();
        let g = self.evaluator.jacobian(features);

        g * self.transform.grad(v).into_vector()
    }

    fn update_grad(&mut self, grad: &Matrix<f64>, update: Self::Output) -> UpdateResult<()> {
        self.evaluator.update_grad(grad, update)
    }

    fn update(&mut self, features: &Features, update: Self::Output) -> UpdateResult<()> {
        match self.evaluator.evaluate(features) {
            Ok(v) => self.evaluator.update(
                features, self.transform.grad(v).elementwise_mul(&update)),
            Err(_) => Err(UpdateError::Failed)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        basis::fixed::Polynomial,
        core::{Approximator, Parameterised, Embedding},
        transforms::Logistic,
    };
    use super::TransformedLFA;

    #[test]
    fn test_logistic_lfa() {
        let mut fa = TransformedLFA::scalar(
            Polynomial::new(2, vec![(-1.0, 1.0)]),
            Logistic::default(),
        );

        for _ in 0..10000 {
            let x = fa.embed(&vec![-1.0]);
            let y_apx = fa.evaluate(&x).unwrap();

            fa.update(&x, -1.0 - y_apx).ok();

            let x = fa.embed(&vec![1.0]);
            let y_apx = fa.evaluate(&x).unwrap();

            fa.update(&x, 1.0 - y_apx).ok();
        }

        for x in -10..10 {
            let v = fa.evaluate(&fa.embed(&vec![x as f64 / 10.0])).unwrap();

            assert!(v > 0.0);
        }
    }
}
