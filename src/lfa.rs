use crate::approximators::{ScalarFunction, VectorFunction};
use crate::basis::{Projection, Projector};
use crate::core::*;
use crate::geometry::{Card, Space, Matrix};
use std::{collections::HashMap, marker::PhantomData};

/// Linear function approximator.
#[derive(Clone, Serialize, Deserialize)]
pub struct LFA<I: ?Sized, P: Projector<I>, A: Approximator<Projection>> {
    projector: P,
    approximator: A,

    phantom: PhantomData<I>,
}

impl<I: ?Sized, P: Projector<I>, A: Approximator<Projection>> LFA<I, P, A> {
    pub fn new(projector: P, approximator: A) -> Self {
        LFA {
            projector: projector,
            approximator: approximator,

            phantom: PhantomData,
        }
    }
}

impl<I: ?Sized, P: Projector<I>> LFA<I, P, ScalarFunction> {
    pub fn scalar_output(projector: P) -> Self {
        let approximator = ScalarFunction::new(projector.dim());

        Self::new(projector, approximator)
    }
}

impl<I: ?Sized, P: Projector<I>> LFA<I, P, VectorFunction> {
    pub fn vector_output(projector: P, n_outputs: usize) -> Self {
        let approximator = VectorFunction::new(projector.dim(), n_outputs);

        Self::new(projector, approximator)
    }
}

impl<I, P: Projector<I>, A: Approximator<Projection>> Space for LFA<I, P, A> {
    type Value = Projection;

    fn dim(&self) -> usize {
        self.projector.dim()
    }

    fn card(&self) -> Card {
        self.projector.card()
    }
}

impl<I, P, A> Projector<I> for LFA<I, P, A>
where
    P: Projector<I>,
    A: Approximator<Projection>,
{
    fn project(&self, input: &I) -> Projection {
        self.projector.project(input)
    }
}

impl<I, P, A> Approximator<I> for LFA<I, P, A>
where
    I: ?Sized,
    P: Projector<I>,
    A: Approximator<Projection>,
{
    type Value = A::Value;

    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Value> {
        let primal = self.projector.project(input);

        self.approximator.evaluate(&primal)
    }

    fn update(&mut self, input: &I, update: Self::Value) -> UpdateResult<()> {
        let primal = self.projector.project(input);

        self.approximator.update(&primal, update)
    }

    fn adapt(&mut self, new_features: &HashMap<IndexT, IndexSet>) -> AdaptResult<usize> {
        self.approximator.adapt(new_features)
    }
}

impl<I, P, A> Parameterised for LFA<I, P, A>
where
    I: ?Sized,
    P: Projector<I>,
    A: Approximator<Projection> + Parameterised,
{
    fn weights(&self) -> Matrix<f64> { self.approximator.weights() }
}
