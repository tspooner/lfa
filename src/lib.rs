//! `lfa` is a framework for online learning of linear function approximation
//! models. Included is a suite of scalar and multi-output approximation
//! helpers, a range of basis functions for feature construction, and a set of
//! optimisation routines. The framework is designed to support
//! fallible operations for which a custom error type is provided.
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
//!     LFA,
//!     basis::{Basis, Polynomial},
//!     optim::SGD,
//! };
//! use rand::{Rng, thread_rng};
//! use rand_distr::{Uniform, Normal};
//! use spaces::Space;
//!
//! fn main() {
//!     const M: f64 = 1.0;
//!     const C: f64 = -0.5;
//!
//!     let mut fa = LFA::scalar(
//!         Polynomial::new(1, 1).with_zeroth(),
//!         SGD(1.0)
//!     );
//!
//!     let mut rng = thread_rng();
//!     let mut data_dist = Uniform::new_inclusive(-1.0, 1.0);
//!     let mut noise_dist = Normal::new(0.0, 0.01).unwrap();
//!
//!     for x in rng.sample_iter(&data_dist).take(1000) {
//!         let y_exp = M*x + C;
//!         let y_sample = y_exp + rng.sample(noise_dist);
//!
//!         fa.update_with(&[x], |_, y_pred| y_sample - y_pred).ok();
//!     }
//!
//!     let rmse = (
//!         (fa.weights[0] - M).powi(2) +
//!         (fa.weights[1] - C).powi(2)
//!     ).sqrt();
//!
//!     assert!(rmse < 1e-3);
//! }
//! ```
// #![warn(missing_docs)]

#[cfg_attr(feature = "serde", macro_use)]
#[cfg(feature = "serde")]
extern crate serde_crate as serde;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck;

mod macros;
mod utils;

#[macro_use]
mod error;

mod features;
pub use self::features::*;

#[macro_use]
pub mod basis;
pub mod optim;

pub use self::error::*;

/// Linear function approximator with scalar output.
pub type ScalarLFA<B, O> = LFA<B, ndarray::Array1<f64>, O>;

/// Linear function approximator with vector output.
pub type VectorLFA<B, O> = LFA<B, ndarray::Array2<f64>, O>;

/// Linear function approximator.
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct LFA<B, W, O = optim::SGD> {
    /// Basis representation used to generate feature vectors.
    pub basis: B,

    /// Weights collection.
    pub weights: W,

    /// Optimisation routine used during updates.
    pub optimiser: O,
}

impl<B, W, O> LFA<B, W, O> {
    /// Construct an arbitrary `LFA` using a given basis representation, initial
    /// collection of weights and optimisation routine.
    pub fn new(basis: B, weights: W, optimiser: O) -> Self {
        LFA {
            basis,
            weights,
            optimiser,
        }
    }
}

impl<B, O> ScalarLFA<B, O> {
    /// Construct an `ScalarLFA` using a given basis representation and
    /// optimisation routine. The weights are initialised with a vector of
    /// zeros.
    pub fn scalar(basis: B, optimiser: O) -> Self
    where
        B: spaces::Space,
    {
        let n: usize = basis.dim().into();
        let weights = ndarray::Array1::zeros(n);

        LFA {
            basis,
            weights,
            optimiser,
        }
    }

    /// Evaluate the function approximator for a given `input`.
    pub fn evaluate<I>(&self, input: I) -> Result<f64>
    where
        B: basis::Basis<I, Value = Features>,
    {
        self.basis
            .project(input)
            .map(|b| b.dot(&self.weights.view()))
    }

    /// Update the function approximator for a given `input` and `error`.
    pub fn update<I>(&mut self, input: I, error: f64) -> Result<()>
    where
        B: basis::Basis<I, Value = Features>,
        O: optim::Optimiser,
    {
        self.basis.project(input).and_then(|ref b| {
            self.optimiser
                .step_scaled(&mut self.weights.view_mut(), b, error)
        })
    }

    /// Update the function approximator for a given `input`, deferring the
    /// error computation to a closure provided by the user.
    pub fn update_with<I>(&mut self, input: I, f: impl Fn(&Features, f64) -> f64) -> Result<()>
    where
        B: basis::Basis<I, Value = Features>,
        O: optim::Optimiser,
    {
        self.basis
            .project(input)
            .map(|b| (b.dot(&self.weights.view()), b))
            .map(|(v, b)| (f(&b, v), b))
            .and_then(|(e, b)| {
                self.optimiser
                    .step_scaled(&mut self.weights.view_mut(), &b, e)
            })
    }
}

/// An iterator over the evaluated columns of a weight matrix.
pub struct OutputIter<'a> {
    basis: Features,
    lanes: ndarray::iter::LanesIter<'a, f64, ndarray::Ix1>,
}

impl<'a> Iterator for OutputIter<'a> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        self.lanes.next().map(|ref c| self.basis.dot(c))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.lanes.size_hint()
    }
}

impl<'a> ExactSizeIterator for OutputIter<'a> {
    fn len(&self) -> usize {
        self.lanes.len()
    }
}

impl<B, O> VectorLFA<B, O> {
    /// Construct an `VectorLFA` with a chosen number of outputs using a given
    /// basis representation and optimisation routine. The weights are
    /// initialised with a matrix of zeros.
    pub fn vector(basis: B, optimiser: O, n_outputs: usize) -> Self
    where
        B: spaces::Space,
    {
        let n: usize = basis.dim().into();
        let weights = ndarray::Array2::zeros((n, n_outputs));

        LFA {
            basis,
            weights,
            optimiser,
        }
    }

    /// Return the dimensionality of the output..
    pub fn n_outputs(&self) -> usize {
        self.weights.ncols()
    }

    /// Evaluate the function approximator for a given `input`.
    pub fn evaluate<I>(&self, input: I) -> Result<ndarray::Array1<f64>>
    where
        B: basis::Basis<I, Value = Features>,
    {
        self.try_iter(input).map(|it| it.collect())
    }

    /// Evaluate the `i`th output of the function approximator for a given
    /// `input`.
    pub fn evaluate_index<I>(&self, input: I, index: usize) -> Result<f64>
    where
        B: basis::Basis<I, Value = Features>,
    {
        self.basis
            .project(input)
            .map(|b| b.dot(&self.weights.column(index)))
    }

    /// Iterate sequentially over the outputs of the function approximator for a
    /// given `input`.
    ///
    /// __Panics__ if the basis computation fails.
    pub fn iter<'a, I>(&'a self, input: I) -> OutputIter<'a>
    where
        B: basis::Basis<I, Value = Features>,
    {
        OutputIter {
            basis: self.basis.project(input).unwrap(),
            lanes: self.weights.columns().into_iter(),
        }
    }

    /// Iterate sequentially over the outputs of the function approximator for a
    /// given `input`.
    pub fn try_iter<'a, I>(&'a self, input: I) -> Result<OutputIter<'a>>
    where
        B: basis::Basis<I, Value = Features>,
    {
        self.basis.project(input).map(move |basis| OutputIter {
            basis,
            lanes: self.weights.columns().into_iter(),
        })
    }

    /// Update the function approximator for a given `input` and `error`.
    pub fn update<I, E>(&mut self, input: I, errors: E) -> Result<()>
    where
        B: basis::Basis<I, Value = Features>,
        O: optim::Optimiser,
        E: IntoIterator<Item = f64>,
    {
        self.basis.project(input).and_then(|ref b| {
            let opt = &mut self.optimiser;

            errors
                .into_iter()
                .zip(self.weights.columns_mut().into_iter())
                .fold(Ok(()), |acc, (e, mut c)| {
                    acc.and(opt.step_scaled(&mut c, b, e))
                })
        })
    }

    /// Update the `i`th output of the function approximator for a given `input`
    /// and `error`.
    pub fn update_index<I>(&mut self, input: I, index: usize, error: f64) -> Result<()>
    where
        B: basis::Basis<I, Value = Features>,
        O: optim::Optimiser,
    {
        self.basis.project(input).and_then(|ref b| {
            self.optimiser
                .step_scaled(&mut self.weights.column_mut(index), b, error)
        })
    }

    /// Update the function approximator for a given `input`, deferring the
    /// error computation to a closure provided by the user.
    pub fn update_with<I>(
        &mut self,
        input: I,
        f: impl Fn(&Features, ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    ) -> Result<()>
    where
        B: basis::Basis<I, Value = Features>,
        O: optim::Optimiser,
    {
        self.basis.project(input).and_then(|ref b| {
            let opt = &mut self.optimiser;
            let values = b.matmul(&self.weights);
            let errors = f(b, values).into_raw_vec();

            errors
                .into_iter()
                .zip(self.weights.columns_mut().into_iter())
                .fold(Ok(()), |acc, (e, mut c)| {
                    acc.and(opt.step_scaled(&mut c, b, e))
                })
        })
    }
}
