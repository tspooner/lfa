use crate::core::*;
use std::collections::HashMap;

/// An interface for function approximators.
pub trait Approximator<I: ?Sized> {
    type Output;

    /// Return the dimensionality of the output value `Approximator::Output`.
    fn n_outputs(&self) -> usize;

    /// Evaluate the function and return its value.
    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Output>;

    /// Update the approximator's estimate for the given input.
    fn update(&mut self, input: &I, update: Self::Output) -> UpdateResult<()>;

    #[allow(unused_variables)]
    /// Adapt the approximator in light of newly discovered features.
    fn adapt(&mut self, new_features: &HashMap<IndexT, IndexSet>) -> AdaptResult<usize> {
        unimplemented!()
    }
}

impl<I: ?Sized, T: Approximator<I>> Approximator<I> for Box<T> {
    type Output = T::Output;

    fn n_outputs(&self) -> usize { (**self).n_outputs() }

    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Output> { (**self).evaluate(input) }

    fn update(&mut self, input: &I, update: Self::Output) -> UpdateResult<()> {
        (**self).update(input, update)
    }

    fn adapt(&mut self, new_features: &HashMap<IndexT, IndexSet>) -> AdaptResult<usize> {
        (**self).adapt(new_features)
    }
}
