/// An interface for differentiable transformations.
pub trait Transform<T: ?Sized> {
    type Output;

    /// Return the value of the transform for input `x`.
    fn transform(&self, x: T) -> Self::Output;

    /// Return the gradient of the transform for input `x`.
    fn grad(&self, x: T) -> T;
}

pub type EndoTransform<T> = Transform<T, Output = T>;

import_all!(tanh);
import_all!(identity);
import_all!(softplus);
import_all!(logistic);
import_all!(exponential);
