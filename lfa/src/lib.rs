//! # LFA
//!
//! LFA is a framework for linear function approximation with gradient descent.
#[allow(unused_imports)]
#[macro_use]
extern crate lfa_derive;
#[doc(hidden)]
pub use self::lfa_derive::*;

#[cfg(feature = "serialize")] extern crate serde;
#[cfg_attr(feature = "serialize", macro_use)]
#[cfg(feature = "serialize")]
extern crate serde_derive;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck;

extern crate spaces;

mod macros;
mod utils;

#[macro_use]
pub mod basis;
pub mod sgd;

import_all!(error);
import_all!(features);
import_all!(approximators);

pub type Weights = ndarray::Array2<f64>;
pub type WeightsView<'a> = ndarray::ArrayView2<'a, f64>;
pub type WeightsViewMut<'a> = ndarray::ArrayViewMut2<'a, f64>;

/// An interface for types parameterised by a matrix of weights.
pub trait Parameterised {
    /// Return a clone of the weights.
    fn weights(&self) -> Weights { self.weights_view().to_owned() }

    /// Return a read-only view of the weights.
    fn weights_view(&self) -> WeightsView;

    /// Return a mutable view of the weights.
    fn weights_view_mut(&mut self) -> WeightsViewMut;

    /// Return the dimensions of the weight matrix.
    fn weights_dim(&self) -> [usize; 2] {
        let (r, c) = self.weights_view().dim();

        [r, c]
    }
}

/// An interface for function approximators.
pub trait Approximator: Parameterised {
    /// The type of value being approximated.
    type Output;

    /// Return the dimensionality of the output value `Approximator::Output`.
    fn n_outputs(&self) -> usize { self.weights_dim()[1] }

    /// Evaluate the approximator and return its value.
    fn evaluate(&self, features: &Features) -> EvaluationResult<Self::Output>;

    fn update(&mut self, features: &Features, errors: Self::Output) -> UpdateResult<()> {
        self.update_with(&mut sgd::SGD(1.0), features, errors)
    }

    fn update_with<O: sgd::Optimiser>(&mut self, optimiser: &mut O, features: &Features, errors: Self::Output) -> UpdateResult<()>;
}

pub trait ScalarApproximator: Approximator<Output = f64> {}

impl<T: Approximator<Output = f64>> ScalarApproximator for T {}

pub trait PairApproximator: Approximator<Output = [f64; 2]> {
    fn evaluate_first(&self, features: &Features) -> EvaluationResult<f64> {
        Ok(features.dot(&self.weights_view().column(0)))
    }

    fn evaluate_second(&self, features: &Features) -> EvaluationResult<f64> {
        Ok(features.dot(&self.weights_view().column(1)))
    }
}

impl<T: Approximator<Output = [f64; 2]>> PairApproximator for T {}

pub trait TripleApproximator: Approximator<Output = [f64; 3]> {
    fn evaluate_first(&self, features: &Features) -> EvaluationResult<f64> {
        Ok(features.dot(&self.weights_view().column(0)))
    }

    fn evaluate_second(&self, features: &Features) -> EvaluationResult<f64> {
        Ok(features.dot(&self.weights_view().column(1)))
    }

    fn evaluate_third(&self, features: &Features) -> EvaluationResult<f64> {
        Ok(features.dot(&self.weights_view().column(2)))
    }
}

impl<T: Approximator<Output = [f64; 3]>> TripleApproximator for T {}

pub trait VectorApproximator: Approximator<Output = Vec<f64>> {
    fn evaluate_index(&self, features: &Features, index: usize) -> EvaluationResult<f64> {
        Ok(features.dot(&self.weights_view().column(index)))
    }

    fn update_index(&mut self, features: &Features, index: usize, update: f64) -> UpdateResult<()> {
        use crate::sgd::Optimiser;

        sgd::SGD(1.0).step(&mut self.weights_view_mut().column_mut(index), features, update)
    }

    fn update_index_with<O: sgd::Optimiser>(&mut self, opt: &mut O, features: &Features, index: usize, update: f64) -> UpdateResult<()> {
        opt.step(&mut self.weights_view_mut().column_mut(index), features, update)
    }
}

impl<T: Approximator<Output = Vec<f64>>> VectorApproximator for T {}
