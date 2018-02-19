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

impl<I: ?Sized, V, T: Function<I, V>> Function<I, V> for Box<T> {
    fn evaluate(&self, input: &I) -> EvaluationResult<V> { (**self).evaluate(input) }
}


pub trait Parameterised<I: ?Sized, V> {
    fn update(&mut self, input: &I, update: V) -> UpdateResult<()>;
}

impl<I: ?Sized, V, T: Parameterised<I, V>> Parameterised<I, V> for Box<T> {
    fn update(&mut self, input: &I, update: V) -> UpdateResult<()> {
        (**self).update(input, update)
    }
}


/// An interface for dealing with function approximators.
pub trait Approximator<I: ?Sized, V>: Function<I, V> + Parameterised<I, V> {}

impl<I: ?Sized, V, T: Approximator<I, V>> Approximator<I, V> for Box<T> {}


pub mod projection;
pub use self::projection::{Projection, Projector};

mod table;
pub use self::table::Table;

mod linear;
pub use self::linear::Linear;
