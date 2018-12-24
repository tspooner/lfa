use crate::approximators::*;
use crate::basis::{Projection, Projector, AdaptiveProjector, CandidateFeature};
use crate::core::*;
use crate::geometry::{Card, Space, Matrix};
use std::{collections::HashMap, marker::PhantomData};

macro_rules! impl_concrete_builder {
    ($ftype:ty => $fname:ident) => {
        impl<I, P> LFA<I, P, $ftype>
        where
            I: ?Sized,
            P: Projector<I>,
        {
            pub fn $fname(projector: P) -> Self {
                let approximator = <$ftype>::new(projector.dim());

                Self::new(projector, approximator)
            }
        }

        impl<I, P> From<P> for LFA<I, P, $ftype>
        where
            I: ?Sized,
            P: Projector<I>,
        {
            fn from(projector: P) -> Self {
                Self::$fname(projector)
            }
        }
    }
}

/// Linear function approximator.
#[derive(Clone, Serialize, Deserialize)]
pub struct LFA<I: ?Sized, P, A> {
    projector: P,
    approximator: A,

    phantom: PhantomData<I>,
}

impl<I, P, A> LFA<I, P, A>
where
    I: ?Sized,
    P: Projector<I>,
    A: Approximator<Projection>,
{
    pub fn new(projector: P, approximator: A) -> Self {
        LFA {
            projector: projector,
            approximator: approximator,

            phantom: PhantomData,
        }
    }
}

impl_concrete_builder!(ScalarFunction => scalar_output);
impl_concrete_builder!(PairFunction => pair_output);
impl_concrete_builder!(TripleFunction => triple_output);

impl<I, P> LFA<I, P, VectorFunction>
where
    I: ?Sized,
    P: Projector<I>,
{
    pub fn vector_output(projector: P, n_outputs: usize) -> Self {
        let approximator = VectorFunction::new(projector.dim(), n_outputs);

        Self::new(projector, approximator)
    }
}

impl<I, P, A> LFA<I, P, A>
where
    I: ?Sized,
    A: Approximator<Projection>,
{
    #[allow(dead_code)]
    fn evaluate_primal(&mut self, primal: &Projection) -> EvaluationResult<A::Value> {
        self.approximator.evaluate(primal)
    }

    #[allow(dead_code)]
    fn update_primal(&mut self, primal: &Projection, update: A::Value) -> UpdateResult<()> {
        self.approximator.update(primal, update)
    }
}

impl<I, P, A> Space for LFA<I, P, A>
where
    I: ?Sized,
    P: Projector<I>,
{
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
{
    fn project(&self, input: &I) -> Projection {
        self.projector.project(input)
    }
}

impl<I, P, A> AdaptiveProjector<I> for LFA<I, P, A>
where
    P: AdaptiveProjector<I>,
{
    fn discover(&mut self, input: &I, error: f64) -> Option<HashMap<IndexT, IndexSet>> {
        self.projector.discover(input, error)
    }

    fn add_feature(&mut self, candidate: CandidateFeature) -> Option<(usize, IndexSet)> {
        self.projector.add_feature(candidate)
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
    A: Parameterised,
{
    fn weights(&self) -> Matrix<f64> { self.approximator.weights() }
}
