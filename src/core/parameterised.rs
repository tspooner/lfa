use crate::geometry::Matrix;

/// An interface for approximators parameterised by a set of weights.
pub trait Parameterised {
    /// Return a copy of the approximator weights.
    fn weights(&self) -> Matrix<f64>;

    /// Returns the total number of weights.
    fn n_weights(&self) -> usize { self.weights().len() }
}

impl<T: Parameterised> Parameterised for Box<T> {
    fn weights(&self) -> Matrix<f64> { (**self).weights() }

    fn n_weights(&self) -> usize { (**self).n_weights() }
}
