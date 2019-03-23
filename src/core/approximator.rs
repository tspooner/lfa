use crate::core::*;

/// An interface for function approximators.
pub trait Approximator<I: ?Sized> {
    type Output;

    /// Return the dimensionality of the output value `Approximator::Output`.
    fn n_outputs(&self) -> usize;

    /// Evaluate the function and return its value.
    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Output>;

    /// Update the approximator's estimate for the given input.
    fn update(&mut self, input: &I, update: Self::Output) -> UpdateResult<()>;
}

impl<I: ?Sized, T: Approximator<I>> Approximator<I> for Box<T> {
    type Output = T::Output;

    fn n_outputs(&self) -> usize { (**self).n_outputs() }

    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Output> { (**self).evaluate(input) }

    fn update(&mut self, input: &I, update: Self::Output) -> UpdateResult<()> {
        (**self).update(input, update)
    }
}

pub trait LinearApproximator<I: ?Sized>: Approximator<I> {
    fn to_primal(&self, input: &I) -> Projection;

    fn evaluate_primal(&self, primal: &Projection) -> EvaluationResult<Self::Output>;

    fn update_primal(&mut self, primal: &Projection, update: Self::Output) -> UpdateResult<()>;
}

impl<I: ?Sized, T: LinearApproximator<I>> LinearApproximator<I> for Box<T> {
    fn to_primal(&self, input: &I) -> Projection {
        (**self).to_primal(input)
    }

    fn evaluate_primal(&self, primal: &Projection) -> EvaluationResult<Self::Output> {
        (**self).evaluate_primal(primal)
    }

    fn update_primal(&mut self, primal: &Projection, update: Self::Output) -> UpdateResult<()> {
        (**self).update_primal(primal, update)
    }
}
