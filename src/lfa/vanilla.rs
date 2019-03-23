use crate::{
    core::*,
    eval::*,
    geometry::{Matrix, Space},
};
use std::collections::HashMap;

macro_rules! impl_builder {
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

impl_builder!(ScalarFunction => scalar);
impl_builder!(PairFunction => pair);
impl_builder!(TripleFunction => triple);

/// Linear function approximator.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LFA<P, E> {
    pub projector: P,
    pub evaluator: E,
}

impl<P, E> LFA<P, E> {
    pub fn new(projector: P, evaluator: E) -> Self {
        LFA { projector, evaluator, }
    }
}

impl<P: Space> LFA<P, VectorFunction> {
    pub fn vector(projector: P, n_outputs: usize) -> Self {
        let evaluator = VectorFunction::zeros(projector.dim(), n_outputs);

        Self::new(projector, evaluator)
    }
}

impl<I: ?Sized, P, E> Approximator<I> for LFA<P, E>
where
    P: Projector<I>,
    E: Approximator<Projection>,
{
    type Output = E::Output;

    fn n_outputs(&self) -> usize {
        self.evaluator.n_outputs()
    }

    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Output> {
        self.evaluate_primal(&self.projector.project(input))
    }

    fn update(&mut self, input: &I, update: Self::Output) -> UpdateResult<()> {
        self.update_primal(&self.projector.project(input), update)
    }
}

impl<I: ?Sized, P, E> LinearApproximator<I> for LFA<P, E>
where
    P: Projector<I>,
    E: Approximator<Projection>,
{
    fn to_primal(&self, input: &I) -> Projection {
        self.projector.project(input)
    }

    fn evaluate_primal(&self, primal: &Projection) -> EvaluationResult<Self::Output> {
        self.evaluator.evaluate(primal)
    }

    fn update_primal(&mut self, primal: &Projection, update: Self::Output) -> UpdateResult<()> {
        self.evaluator.update(primal, update)
    }
}

impl<P, E: Parameterised> Parameterised for LFA<P, E> {
    fn weights(&self) -> Matrix<f64> { self.evaluator.weights() }

    fn n_weights(&self) -> usize { self.evaluator.n_weights() }
}
