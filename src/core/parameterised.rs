use crate::geometry::Matrix;

/// An interface for approximators parameterised by a set of weights.
pub trait Parameterised {
    /// Return a copy of the approximator weights.
    fn weights(&self) -> Matrix<f64>;
}

impl<T: Parameterised> Parameterised for Box<T> {
    fn weights(&self) -> Matrix<f64> { (**self).weights() }
}
