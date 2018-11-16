use approximators::*;
use core::*;
use geometry::Matrix;
use projectors::{IndexSet, IndexT, Projection, Projector};
use std::{collections::HashMap, marker::PhantomData};

/// Linear Basis Function Model
#[derive(Clone, Serialize, Deserialize)]
pub struct LBFM<I: ?Sized, P: Projector<I>, A: Approximator<Projection>> {
    pub projector: P,
    approximator: A,

    phantom: PhantomData<I>,
}

impl<I: ?Sized, P: Projector<I>, A: Approximator<Projection>> LBFM<I, P, A> {
    fn new(projector: P, approximator: A) -> Self {
        LBFM {
            projector: projector,
            approximator: approximator,

            phantom: PhantomData,
        }
    }
}

impl<I: ?Sized, P: Projector<I>> LBFM<I, P, ScalarFunction> {
    pub fn scalar_valued(projector: P) -> Self {
        let approximator = ScalarFunction::new(projector.dim());

        Self::new(projector, approximator)
    }
}

impl<I: ?Sized, P: Projector<I>> LBFM<I, P, VectorFunction> {
    pub fn vector_valued(projector: P, n_outputs: usize) -> Self {
        let approximator = VectorFunction::new(projector.dim(), n_outputs);

        Self::new(projector, approximator)
    }
}

impl<I: ?Sized, P: Projector<I>, A: Approximator<Projection>> Approximator<I> for LBFM<I, P, A> {
    type Value = A::Value;

    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Value> {
        self.approximator.evaluate(&self.projector.project(input))
    }

    fn update(&mut self, input: &I, update: Self::Value) -> UpdateResult<()> {
        self.approximator
            .update(&self.projector.project(input), update)
    }

    fn adapt(&mut self, new_features: &HashMap<IndexT, IndexSet>) -> AdaptResult<usize> {
        self.approximator.adapt(new_features)
    }
}

impl<I, P, A> Parameterised for LBFM<I, P, A>
where
    I: ?Sized,
    P: Projector<I>,
    A: Approximator<Projection> + Parameterised,
{
    fn weights(&self) -> Matrix<f64> { self.approximator.weights() }
}
