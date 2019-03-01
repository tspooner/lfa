use crate::core::Gradient;

/// An interface for differentiable transformations.
pub trait Transform<T> {
    /// Return the value of the transform for input `x`.
    fn transform(&self, x: T) -> T;

    /// Return the gradient of the transform for input `x`.
    fn grad(&self, x: T) -> T where T: Gradient;
}

import_all!(identity);
import_all!(softplus);
import_all!(logistic);
import_all!(exponential);
