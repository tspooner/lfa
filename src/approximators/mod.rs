use error::*;

mod simple;
pub use self::simple::Simple;

mod multi;
pub use self::multi::Multi;

/// An interface for function approximators.
pub trait Approximator<I: ?Sized> {
    type Value;

    /// Evaluate the function and return its value.
    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Value>;

    /// Update the approximator's estimate for the given input.
    fn update(&mut self, input: &I, update: Self::Value) -> UpdateResult<()>;
}

impl<I: ?Sized, T: Approximator<I>> Approximator<I> for Box<T> {
    type Value = T::Value;

    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Value> { (**self).evaluate(input) }

    fn update(&mut self, input: &I, update: Self::Value) -> UpdateResult<()> {
        (**self).update(input, update)
    }
}

/// An interface for adaptive function approximators.
pub trait AdaptiveApproximator<I: ?Sized>: Approximator<I> {
    /// Adapt the approximator given some approximation error.
    fn adapt(&mut self, input: &I, error: Self::Value) -> AdaptResult<usize>;
}
