use crate::{
    basis::{Projection, Projector},
    core::*,
    eval::*,
    geometry::{Matrix, Space},
    transforms::{Transform, Identity},
};
use std::collections::HashMap;

macro_rules! impl_concrete_builder {
    ($ftype:ty => $fname:ident) => {
        impl<P: Space> LFA<P, $ftype> {
            pub fn $fname(projector: P) -> Self {
                let evaluator = <$ftype>::zeros(projector.dim());

                Self::new(projector, evaluator)
            }
        }

        impl<P: Space> From<P> for LFA<P, $ftype> {
            fn from(projector: P) -> Self { Self::$fname(projector) }
        }
    };
}

/// Linear function approximator.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LFA<P, E, T = Identity> {
    pub projector: P,
    pub evaluator: E,
    pub transform: T,
}

impl<P, E> LFA<P, E> {
    pub fn new(projector: P, evaluator: E) -> Self {
        LFA { projector, evaluator, transform: Identity, }
    }
}

impl_concrete_builder!(ScalarFunction => scalar);
impl_concrete_builder!(PairFunction => pair);
impl_concrete_builder!(TripleFunction => triple);

impl<P: Space> LFA<P, VectorFunction> {
    pub fn vector(projector: P, n_outputs: usize) -> Self {
        let evaluator = VectorFunction::zeros(projector.dim(), n_outputs);

        Self::new(projector, evaluator)
    }
}

impl<P, E, T> LFA<P, E, T>
where
    E: Approximator<Projection>,
    T: Transform<E::Value>,
{
    #[allow(dead_code)]
    pub fn evaluate_primal(&self, primal: &Projection) -> EvaluationResult<E::Value> {
        self.evaluator.evaluate(primal).map(|v| self.transform.transform(v))
    }

    #[allow(dead_code)]
    pub fn update_primal(&mut self, primal: &Projection, update: E::Value) -> UpdateResult<()>
        where E::Value: Gradient
    {
        let value = self.evaluate_primal(primal).unwrap();

        self.evaluator.update(primal, self.transform.grad(value).chain(update))
    }
}

impl<I: ?Sized, P, E, T> Approximator<I> for LFA<P, E, T>
where
    P: Projector<I>,
    E: Approximator<Projection>,
    T: Transform<E::Value>,
    E::Value: Gradient,
{
    type Value = E::Value;

    fn n_outputs(&self) -> usize {
        self.evaluator.n_outputs()
    }

    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Value> {
        self.evaluate_primal(&self.projector.project(input))
    }

    fn update(&mut self, input: &I, update: Self::Value) -> UpdateResult<()> {
        self.update_primal(&self.projector.project(input), update)
    }

    fn adapt(&mut self, new_features: &HashMap<IndexT, IndexSet>) -> AdaptResult<usize> {
        self.evaluator.adapt(new_features)
    }
}

impl<P, E: Parameterised, T> Parameterised for LFA<P, E, T> {
    fn weights(&self) -> Matrix<f64> { self.evaluator.weights() }

    fn n_weights(&self) -> usize { self.evaluator.n_weights() }
}
