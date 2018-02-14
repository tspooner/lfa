extern crate serde;
#[macro_use]
extern crate serde_derive;

extern crate spaces as geometry;

mod utils;

mod error;
pub use self::error::*;

pub trait Function<I: ?Sized, V> {
    /// Evaluates the function and returns its output.
    fn evaluate(&self, input: &I) -> EvaluationResult<V>;
}

impl<I: ?Sized, V, T> Function<I, V> for Box<T>
where T: Function<I, V>
{
    fn evaluate(&self, input: &I) -> EvaluationResult<V> { (**self).evaluate(input) }
}

pub trait Parameterised<I: ?Sized, V> {
    fn update(&mut self, input: &I, update: V) -> UpdateResult<()>;
}

impl<I: ?Sized, V, T> Parameterised<I, V> for Box<T>
where T: Parameterised<I, V>
{
    fn update(&mut self, input: &I, update: V) -> UpdateResult<()> {
        (**self).update(input, update)
    }
}

/// An interface for dealing with function approximators.
pub trait Approximator<I: ?Sized, V>: Function<I, V> + Parameterised<I, V> {}

impl<I: ?Sized, V, T> Approximator<I, V> for Box<T> where T: Approximator<I, V> {}

pub mod projection;
pub use self::projection::{Projection, Projector};

mod table;
pub use self::table::Table;

mod linear;
pub use self::linear::Linear;
