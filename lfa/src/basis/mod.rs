//! Basis function representations used to generate [`Features`].
//!
//! [`Features`]: enum.Features.html
use crate::{ActivationT, Features, IndexT, Result};
use itertools::Itertools;
use spaces::{real::Interval, BoundedSpace};

pub mod kernels;

/// Trait for functional bases.
pub trait Basis {
    /// Return the number of features produced by this [`Basis`].
    ///
    /// [`Basis`]: trait.Basis.html
    fn n_features(&self) -> usize;

    /// Compute the the i<sup>th</sup> element of the projection for the given `input`.
    ///
    /// __Note:__ in most cases this can be computed without a full projection, but this is _not
    /// guaranteed_.
    fn project_ith(&self, input: &[f64], index: IndexT) -> Result<Option<ActivationT>>;

    /// Project the given `input` onto the basis.
    ///
    /// __Note:__ by default this method is implemented using the [`project_ith`] method.
    ///
    /// ```
    /// use lfa::{Features, basis::{Basis, Constant}};
    ///
    /// let basis = Constant::unit();
    ///
    /// assert_eq!(
    ///     basis.project(&[0.0]).unwrap(),
    ///     Features::from(vec![1.0])
    /// );
    /// ```
    ///
    /// [`project_ith`]: #tymethod.project_ith
    fn project(&self, input: &[f64]) -> Result<Features> {
        (0..self.n_features())
            .into_iter()
            .map(|i| self.project_ith(input, i).map(|x| x.unwrap_or(0.0)))
            .collect()
    }

    /// Return a stack of this [`Basis`] over another.
    ///
    /// [`Basis`]: trait.Basis.html
    fn stack<P: Basis>(self, other: P) -> Stacker<Self, P>
    where Self: Sized {
        Stacker::new(self, other)
    }

    /// Return the a stack of this [`Basis`] with a single constant feature
    /// term.
    ///
    /// [`Basis`]: trait.Basis.html
    fn with_constant(self) -> Stacker<Self, Constant>
    where Self: Sized {
        self.stack(Constant::unit())
    }

    /// Return the original [`Basis`] with all activations normalised in
    /// _L₀_.
    ///
    /// [`Basis`]: trait.Basis.html
    fn normalise_l0(self) -> L0Normaliser<Self>
    where Self: Sized {
        L0Normaliser::new(self)
    }

    /// Return the original [`Basis`] with all activations normalised in
    /// _L₁_.
    ///
    /// [`Basis`]: trait.Basis.html
    fn normalise_l1(self) -> L1Normaliser<Self>
    where Self: Sized {
        L1Normaliser::new(self)
    }

    /// Return the original [`Basis`] with all activations normalised in
    /// _L₂_.
    ///
    /// [`Basis`]: trait.Basis.html
    fn normalise_l2(self) -> L2Normaliser<Self>
    where Self: Sized {
        L2Normaliser::new(self)
    }

    /// Return the original [`Basis`] with all activations normalised in
    /// _L∞_.
    ///
    /// [`Basis`]: trait.Basis.html
    fn normalise_linf(self) -> LinfNormaliser<Self>
    where Self: Sized {
        LinfNormaliser::new(self)
    }
}

import_all!(stack);
import_all!(normalisation);

import_all!(fourier);
import_all!(polynomial);
import_all!(tile_coding);
import_all!(kernelised);
import_all!(uniform_grid);
import_all!(constant);

#[cfg(feature = "random")]
import_all!(random);

pub(self) fn compute_coefficients(order: u8, dim: usize) -> impl Iterator<Item = Vec<u8>> {
    (0..dim)
        .map(|_| 0..=order)
        .multi_cartesian_product()
        .skip(1)
        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
        .dedup()
}

pub(self) fn get_bounds(d: &Interval) -> (f64, f64) {
    match (d.inf(), d.sup()) {
        (Some(lb), Some(ub)) => (lb, ub),
        (Some(_), None) => panic!("Dimension {} is missing an upper bound (sup).", d),
        (None, Some(_)) => panic!("Dimension {} is missing a lower bound (inf).", d),
        (None, None) => panic!("Dimension {} must be bounded.", d),
    }
}
