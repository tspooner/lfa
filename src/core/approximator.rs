use crate::{
    core::*,
    geometry::{Vector, Matrix, MatrixView, MatrixViewMut},
};

pub type Features = Projection;

/// An interface for types parameterised by a matrix of weights.
pub trait Parameterised {
    /// Return a clone of the weights.
    fn weights(&self) -> Matrix<f64> { self.weights_view().to_owned() }

    /// Return a read-only view of the weights.
    fn weights_view(&self) -> MatrixView<f64>;

    /// Return a mutable view of the weights.
    fn weights_view_mut(&mut self) -> MatrixViewMut<f64>;

    /// Return the dimensions of the weight matrix.
    fn weights_dim(&self) -> (usize, usize) {
        self.weights_view().dim()
    }

    /// Return the total number of weights.
    fn weights_count(&self) -> usize {
        self.weights_view().len()
    }
}

/// An interface for types with an embedded feature representation.
pub trait Embedded<I: ?Sized> {
    /// Return the number of features in the representation.
    fn n_features(&self) -> usize;

    /// Convert some input `I` into a `Features` vector.
    fn to_features(&self, input: &I) -> Features;
}

/// An interface for function approximators.
pub trait Approximator {
    /// The value being approximated.
    type Output;

    /// Return the dimensionality of the output value `Approximator::Output`.
    fn n_outputs(&self) -> usize;

    /// Evaluate the approximator and return its value.
    fn evaluate(&self, features: &Features) -> EvaluationResult<Self::Output>;

    /// Update the approximator's estimate for the given input.
    fn update(&mut self, features: &Features, update: Self::Output) -> UpdateResult<()>;
}

// TODO: Implement more efficient variants for LFA types when impl specialisation is released on
// stable.
pub trait ScalarApproximator: Approximator<Output = f64> {}

impl<T: Approximator<Output = f64>> ScalarApproximator for T {}

pub trait PairApproximator: Approximator<Output = (f64, f64)> {
    fn evaluate_first(&self, features: &Features) -> EvaluationResult<f64> {
        self.evaluate(features).map(|v| v.0)
    }

    fn evaluate_second(&self, features: &Features) -> EvaluationResult<f64> {
        self.evaluate(features).map(|v| v.1)
    }
}

impl<T: Approximator<Output = (f64, f64)>> PairApproximator for T {}

pub trait TripleApproximator: Approximator<Output = (f64, f64, f64)> {
    fn evaluate_first(&self, features: &Features) -> EvaluationResult<f64> {
        self.evaluate(features).map(|v| v.0)
    }

    fn evaluate_second(&self, features: &Features) -> EvaluationResult<f64> {
        self.evaluate(features).map(|v| v.1)
    }

    fn evaluate_third(&self, features: &Features) -> EvaluationResult<f64> {
        self.evaluate(features).map(|v| v.2)
    }
}

impl<T: Approximator<Output = (f64, f64, f64)>> TripleApproximator for T {}

pub trait VectorApproximator: Approximator<Output = Vector<f64>> {
    fn evaluate_index(&self, features: &Features, index: usize) -> EvaluationResult<f64> {
        self.evaluate(features).map(|v| v[index])
    }

    fn update_index(&mut self, features: &Features, index: usize, update: f64) -> UpdateResult<()> {
        let mut update_vec = vec![0.0; self.n_outputs()];
        update_vec[index] = update;

        self.update(features, Vector::from_vec(update_vec))
    }
}

impl<T: Approximator<Output = Vector<f64>>> VectorApproximator for T {}
