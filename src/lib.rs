extern crate serde;
#[macro_use]
extern crate serde_derive;

extern crate spaces as geometry;

mod utils;

mod error;
pub use self::error::*;


/// An interface for dealing with function approximators.
pub trait Approximator<I: ?Sized> {
    type Value;

    /// Evaluates the function and returns its output.
    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Value>;

    fn update(&mut self, input: &I, update: Self::Value) -> UpdateResult<()>;
}

impl<I: ?Sized, T: Approximator<I>> Approximator<I> for Box<T> {
    type Value = T::Value;

    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Value> { (**self).evaluate(input) }

    fn update(&mut self, input: &I, update: Self::Value) -> UpdateResult<()> {
        (**self).update(input, update)
    }
}


pub mod projection;
pub use self::projection::{Projection, Projector};

mod approximators;
pub use self::approximators::*;
