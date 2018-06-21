use approximators::{Multi, Simple};
use core::{Approximator, Parameterised};
use error::*;
use geometry::Matrix;
use projectors::{IndexSet, IndexT, Projection, Projector};
use std::{collections::HashMap, marker::PhantomData};

#[derive(Clone, Serialize, Deserialize)]
pub struct LFA<I: ?Sized, P: Projector<I>, A: Approximator<Projection>> {
    pub projector: P,
    pub approximator: A,

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

impl<I: ?Sized, P: Projector<I>> LFA<I, P, Simple> {
    pub fn simple(projector: P) -> Self {
        LFA {
            approximator: Simple::new(projector.dim()),
            projector: projector,

            phantom: PhantomData,
        }
    }
}

impl<I: ?Sized, P: Projector<I>> LFA<I, P, Multi> {
    pub fn multi(projector: P, n_outputs: usize) -> Self {
        LFA {
            approximator: Multi::new(projector.dim(), n_outputs),
            projector: projector,

            phantom: PhantomData,
        }
    }
}

impl<I: ?Sized, P: Projector<I>, A: Approximator<Projection>> Approximator<I> for LFA<I, P, A> {
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

impl<I, P, A> Parameterised for LFA<I, P, A>
where
    I: ?Sized,
    P: Projector<I>,
    A: Approximator<Projection> + Parameterised,
{
    fn weights(&self) -> Matrix<f64> {
        self.approximator.weights()
    }
}
