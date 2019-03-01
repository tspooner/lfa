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
    pub projector: P,
    pub approximator: A,
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
    pub fn evaluate_primal(&self, primal: &Projection) -> EvaluationResult<A::Value> {
        self.approximator.evaluate(primal)
    }

    #[allow(dead_code)]
    pub fn update_primal(&mut self, primal: &Projection, update: A::Value) -> UpdateResult<()> {
        self.approximator.update(primal, update)
    }
}

impl<I, P, A> Approximator<I> for LFA<P, A>
where
    I: ?Sized,
    P: Projector<I>,
    A: Approximator<Projection>,
{
    type Value = A::Value;

    fn n_outputs(&self) -> usize {
        self.approximator.n_outputs()
    }

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

impl<P, A: Parameterised> Parameterised for LFA<P, A> {
    fn weights(&self) -> Matrix<f64> { self.approximator.weights() }
}
