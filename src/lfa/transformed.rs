use crate::{
    core::*,
    eval::*,
    geometry::{Matrix, Space},
    transforms::{Transform, Identity},
};
use std::collections::HashMap;

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

impl_builder!(ScalarFunction => scalar);
impl_builder!(PairFunction => pair);
impl_builder!(TripleFunction => triple);

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

impl<P: Space, T> TransformedLFA<P, VectorFunction, T> {
    pub fn vector(projector: P, n_outputs: usize, transform: T) -> Self {
        let evaluator = VectorFunction::zeros(projector.dim(), n_outputs);

        Self::new(projector, evaluator, transform)
    }
}

impl<P, E, T> TransformedLFA<P, E, T>
where
    E: Approximator<Projection>,
    T: Transform<E::Output>,
{
    fn evaluate_primal(&self, primal: &Projection) -> EvaluationResult<E::Output> {
        self.evaluator.evaluate(primal).map(|v| self.transform.transform(v))
    }
}

impl<I: ?Sized, P, E, T> Approximator<I> for TransformedLFA<P, E, T>
where
    P: Projector<I>,
    E: Approximator<Projection>,
    T: Transform<E::Output>,
    E::Output: Gradient,
{
    type Output = E::Output;

    fn n_outputs(&self) -> usize {
        self.evaluator.n_outputs()
    }

    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Output> {
        self.evaluate_primal(&self.projector.project(input))
    }

    fn update(&mut self, input: &I, update: Self::Output) -> UpdateResult<()> {
        let primal = self.projector.project(input);
        let value = self.evaluate_primal(&primal).unwrap();

        self.evaluator.update(&primal, self.transform.grad(value).chain(update))
    }
}

impl<P, E: Parameterised, T> Parameterised for TransformedLFA<P, E, T> {
    fn weights(&self) -> Matrix<f64> { self.evaluator.weights() }

    fn n_weights(&self) -> usize { self.evaluator.n_weights() }
}
