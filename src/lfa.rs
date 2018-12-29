use crate::approximators::*;
use crate::basis::{AdaptiveProjector, CandidateFeature, Projection, Projector};
use crate::core::*;
use crate::geometry::{Card, Matrix, Space};
use std::collections::HashMap;

macro_rules! impl_concrete_builder {
    ($ftype:ty => $fname:ident) => {
        impl<P: Space> LFA<P, $ftype> {
            pub fn $fname(projector: P) -> Self {
                let approximator = <$ftype>::new(projector.dim());

                Self::new(projector, approximator)
            }
        }

        impl<P: Space> From<P> for LFA<P, $ftype> {
            fn from(projector: P) -> Self { Self::$fname(projector) }
        }
    };
}

/// Linear function approximator.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LFA<P, A> {
    projector: P,
    approximator: A,
}

impl<P, A> LFA<P, A> {
    pub fn new(projector: P, approximator: A) -> Self {
        LFA {
            projector: projector,
            approximator: approximator,
        }
    }
}

impl_concrete_builder!(ScalarFunction => scalar_output);
impl_concrete_builder!(PairFunction => pair_output);
impl_concrete_builder!(TripleFunction => triple_output);

impl<P: Space> LFA<P, VectorFunction> {
    pub fn vector_output(projector: P, n_outputs: usize) -> Self {
        let approximator = VectorFunction::new(projector.dim(), n_outputs);

        Self::new(projector, approximator)
    }
}

impl<P, A: Approximator<Projection>> LFA<P, A> {
    #[allow(dead_code)]
    fn evaluate_primal(&mut self, primal: &Projection) -> EvaluationResult<A::Value> {
        self.approximator.evaluate(primal)
    }

    #[allow(dead_code)]
    fn update_primal(&mut self, primal: &Projection, update: A::Value) -> UpdateResult<()> {
        self.approximator.update(primal, update)
    }
}

impl<P: Space, A> Space for LFA<P, A> {
    type Value = Projection;

    fn dim(&self) -> usize { self.projector.dim() }

    fn card(&self) -> Card { self.projector.card() }
}

impl<I, P, A> Projector<I> for LFA<P, A>
where
    I: ?Sized,
    P: Projector<I>,
{
    fn project(&self, input: &I) -> Projection { self.projector.project(input) }
}

impl<I, P, A> AdaptiveProjector<I> for LFA<P, A>
where
    I: ?Sized,
    P: AdaptiveProjector<I>,
{
    fn discover(&mut self, input: &I, error: f64) -> Option<HashMap<IndexT, IndexSet>> {
        self.projector.discover(input, error)
    }

    fn add_feature(&mut self, candidate: CandidateFeature) -> Option<(usize, IndexSet)> {
        self.projector.add_feature(candidate)
    }
}

impl<I, P, A> Approximator<I> for LFA<P, A>
where
    I: ?Sized,
    P: Projector<I>,
    A: Approximator<Projection>,
{
    type Value = A::Value;

    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Value> {
        let primal = self.project(input);

        self.approximator.evaluate(&primal)
    }

    fn update(&mut self, input: &I, update: Self::Value) -> UpdateResult<()> {
        let primal = self.project(input);

        self.approximator.update(&primal, update)
    }

    fn adapt(&mut self, new_features: &HashMap<IndexT, IndexSet>) -> AdaptResult<usize> {
        self.approximator.adapt(new_features)
    }
}

impl<P, A: Parameterised> Parameterised for LFA<P, A> {
    fn weights(&self) -> Matrix<f64> { self.approximator.weights() }
}
