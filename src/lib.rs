extern crate serde;
#[macro_use]
extern crate serde_derive;

extern crate spaces as geometry;

mod utils;

mod error;
pub use self::error::*;

/// An interface for dealing with function approximators.
pub trait Approximator<I: ?Sized, V> {

    /// Evaluates the function and returns its output.
    fn evaluate(&self, input: &I) -> EvaluationResult<V>;

    fn update(&mut self, input: &I, update: V) -> UpdateResult<()>;
}

impl<I: ?Sized, V, T> Approximator<I, V> for Box<T>
where T: Approximator<I, V>
{
    fn evaluate(&self, input: &I) -> EvaluationResult<V> { (**self).evaluate(input) }

    fn update(&mut self, input: &I, update: V) -> UpdateResult<()> {
        (**self).update(input, update)
    }
}

pub mod projection;
pub use self::projection::{Projection, Projector};

mod table;
pub use self::table::Table;

mod linear;
pub use self::linear::Linear;
