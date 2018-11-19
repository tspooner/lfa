use basis::{IndexSet, IndexT};
use core::error::*;
use std::collections::HashMap;

/// An interface for function approximators.
pub trait Approximator<I: ?Sized> {
    type Value;

    /// Evaluate the function and return its value.
    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Value>;

    /// Update the approximator's estimate for the given input.
    fn update(&mut self, input: &I, update: Self::Value) -> UpdateResult<()>;

    #[allow(unused_variables)]
    /// Adapt the approximator in light of newly discovered features.
    fn adapt(&mut self, new_features: &HashMap<IndexT, IndexSet>) -> AdaptResult<usize> {
        Err(AdaptError::NotImplemented)
    }
}

impl<I: ?Sized, T: Approximator<I>> Approximator<I> for Box<T> {
    type Value = T::Value;

    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Value> {
        (**self).evaluate(input)
    }

    fn update(&mut self, input: &I, update: Self::Value) -> UpdateResult<()> {
        (**self).update(input, update)
    }

    fn adapt(&mut self, new_features: &HashMap<IndexT, IndexSet>) -> AdaptResult<usize> {
        (**self).adapt(new_features)
    }
}
