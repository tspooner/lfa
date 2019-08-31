//! `lfa` is a framework for online learning of linear function approximation
//! models. Included is a suite of scalar and multi-output approximator helpers,
//! a range of basis functions for feature construction, and a set of
//! optimisation routines. The framework is also designed to support
//! fallible operations for which a custom error type is provided.
//!
//! # Conventions
//! `lfa` adopts a few conventions to keep in mind:
//!
//! * __[`Weights`](type.Weights.html)__ are restricted to _2-dimensional tensors_ based on the
//! [`ndarray`](https://crates.io/crates/ndarray) crate.
//! * __Rows__ are associated with _features_.
//! * __Columns__ are associated with _outputs_.
//!
//! # Example
//! The code below illustrates how one might use `lfa` to fit a simple linear
//! model with SGD.
//!
//! ```rust,no_run
//! extern crate lfa;
//! extern crate rand;
//! extern crate rand_distr;
//!
//! use lfa::{
//!     Approximator, Parameterised, ScalarFunction,
//!     basis::{Projector, Polynomial},
//!     optim::SGD,
//! };
//! use rand::{Rng, thread_rng};
//! use rand_distr::{Uniform, Normal};
//!
//! fn main() {
//!     const M: f64 = 1.0;
//!     const C: f64 = -0.5;
//!
//!     let basis = Polynomial::new(1, 1).with_constant();
//!
//!     let mut fa = ScalarFunction::zeros(basis.n_features());
//!     let mut opt = SGD(0.01);
//!
//!     let mut rng = thread_rng();
//!     let mut data = Uniform::new_inclusive(-1.0, 1.0);
//!     let mut noise = Normal::new(0.0, 0.01).unwrap();
//!
//!     for x in rng.sample_iter(&data).take(10000) {
//!         let y_exp = M*x + C;
//!         let y_noisy = y_exp + rng.sample(noise);
//!
//!         let x = basis.project(&vec![x]).unwrap();
//!         let y_apx = fa.evaluate(&x).unwrap();
//!
//!         fa.update(&mut opt, &x, y_noisy - y_apx).ok();
//!     }
//!
//!     let weights = fa.weights().column(0).to_owned();
//!     let rmse = (
//!         (weights[0] - M).powi(2) +
//!         (weights[1] - C).powi(2)
//!     ).sqrt();
//!
//!     assert!(rmse < 1e-3);
//! }
//! ```
// #![warn(missing_docs)]

#[allow(unused_imports)]
#[macro_use]
extern crate lfa_derive;
#[doc(hidden)]
pub use self::lfa_derive::*;

#[cfg_attr(feature = "serialize", macro_use)]
#[cfg(feature = "serialize")]
extern crate serde;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck;

extern crate spaces;

mod macros;
mod utils;

#[macro_use]
pub mod basis;
pub mod error;
pub mod optim;

use self::error::*;

import_all!(features);
import_all!(approximators);

/// Matrix populated with _owned_ weights.
pub type Weights = ndarray::Array2<f64>;

/// Matrix populated with _referenced_ weights.
pub type WeightsView<'a> = ndarray::ArrayView2<'a, f64>;

/// Matrix populated with _mutably referenced_ weights.
pub type WeightsViewMut<'a> = ndarray::ArrayViewMut2<'a, f64>;

/// Types that are parameterised by a matrix of weights.
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

/// Function approximators based on [`Features`] representations.
///
/// [`Features`]: enum.Features.html
pub trait Approximator: Parameterised {
    /// The type of value being approximated.
    type Output;

    /// Return the dimensionality of the output value `Approximator::Output`.
    fn n_outputs(&self) -> usize { self.weights_dim()[1] }

    /// Evaluate the approximator and return its value.
    fn evaluate(&self, features: &Features) -> Result<Self::Output>;

    /// Update the approximator using some arbitrary optimiser for a given
    /// error(s).
    fn update<O: optim::Optimiser>(
        &mut self,
        optimiser: &mut O,
        features: &Features,
        errors: Self::Output,
    ) -> Result<()>;
}

/// [`Approximator`]s with an `f64` output.
///
/// [`Approximator`]: trait.Approximator.html
pub trait ScalarApproximator: Approximator<Output = f64> {}

impl<T: Approximator<Output = f64>> ScalarApproximator for T {}

/// [`Approximator`]s with an `[f64; 2]` output.
///
/// [`Approximator`]: trait.Approximator.html
pub trait PairApproximator: Approximator<Output = [f64; 2]> {
    /// Evaluate the first output value.
    fn evaluate_first(&self, features: &Features) -> Result<f64> {
        Ok(features.dot(&self.weights_view().column(0)))
    }

    /// Evaluate the second output value.
    fn evaluate_second(&self, features: &Features) -> Result<f64> {
        Ok(features.dot(&self.weights_view().column(1)))
    }

    /// Update the first output value.
    fn update_first<O: optim::Optimiser>(
        &mut self,
        optimiser: &mut O,
        features: &Features,
        error: f64,
    ) -> Result<()>
    {
        optimiser.step(&mut self.weights_view_mut().column_mut(0), features, error)
    }

    /// Update the second output value.
    fn update_second<O: optim::Optimiser>(
        &mut self,
        optimiser: &mut O,
        features: &Features,
        error: f64,
    ) -> Result<()>
    {
        optimiser.step(&mut self.weights_view_mut().column_mut(1), features, error)
    }
}

impl<T: Approximator<Output = [f64; 2]>> PairApproximator for T {}

/// [`Approximator`]s with an `[f64; 3]` output.
///
/// [`Approximator`]: trait.Approximator.html
pub trait TripleApproximator: Approximator<Output = [f64; 3]> {
    /// Evaluate the first output value.
    fn evaluate_first(&self, features: &Features) -> Result<f64> {
        Ok(features.dot(&self.weights_view().column(0)))
    }

    /// Evaluate the second output value.
    fn evaluate_second(&self, features: &Features) -> Result<f64> {
        Ok(features.dot(&self.weights_view().column(1)))
    }

    /// Evaluate the third output value.
    fn evaluate_third(&self, features: &Features) -> Result<f64> {
        Ok(features.dot(&self.weights_view().column(2)))
    }

    /// Update the first output value.
    fn update_first<O: optim::Optimiser>(
        &mut self,
        optimiser: &mut O,
        features: &Features,
        error: f64,
    ) -> Result<()>
    {
        optimiser.step(&mut self.weights_view_mut().column_mut(0), features, error)
    }

    /// Update the second output value.
    fn update_second<O: optim::Optimiser>(
        &mut self,
        optimiser: &mut O,
        features: &Features,
        error: f64,
    ) -> Result<()>
    {
        optimiser.step(&mut self.weights_view_mut().column_mut(1), features, error)
    }

    /// Update the third output value.
    fn update_third<O: optim::Optimiser>(
        &mut self,
        optimiser: &mut O,
        features: &Features,
        error: f64,
    ) -> Result<()>
    {
        optimiser.step(&mut self.weights_view_mut().column_mut(2), features, error)
    }
}

impl<T: Approximator<Output = [f64; 3]>> TripleApproximator for T {}

/// [`Approximator`]s with an `Vec<f64>` output.
///
/// [`Approximator`]: trait.Approximator.html
pub trait VectorApproximator: Approximator<Output = Vec<f64>> {
    /// Evaluate the `index`-th output value.
    fn evaluate_index(&self, features: &Features, index: usize) -> Result<f64> {
        Ok(features.dot(&self.weights_view().column(index)))
    }

    /// Update the `index`-th output value.
    fn update_index<O: optim::Optimiser>(
        &mut self,
        opt: &mut O,
        features: &Features,
        index: usize,
        error: f64,
    ) -> Result<()>
    {
        opt.step(
            &mut self.weights_view_mut().column_mut(index),
            features,
            error,
        )
    }
}

impl<T: Approximator<Output = Vec<f64>>> VectorApproximator for T {}
