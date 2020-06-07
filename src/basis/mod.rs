//! Basis function representations used to generate [`Features`].
//!
//! [`Features`]: enum.Features.html
use crate::{ActivationT, IndexT, Result};
use itertools::Itertools;
use spaces::{real::Interval, BoundedSpace, Space};
use std::ops::Index;

macro_rules! rescale {
    ($x:ident into $limits:expr) => {
        $x.into_iter().zip($limits.iter()).map(move |(v, l)| {
            let v = v.borrow();

            (v - l.0) / (l.1 - l.0)
        })
    };
}

pub mod kernels;

/// Trait for functional bases.
pub trait Basis<T>: Space + Combinators {
    /// Return the number of features produced by this [`Basis`].
    ///
    /// [`Basis`]: trait.Basis.html
    fn n_features(&self) -> usize { self.dim().into() }

    /// Project the given `input` onto the basis.
    ///
    /// ```
    /// use lfa::{Features, basis::{Basis, Bias}};
    ///
    /// let basis = Bias::unit();
    ///
    /// assert_eq!(
    ///     basis.project(&[0.0]).unwrap(),
    ///     Features::from(vec![1.0])
    /// );
    /// ```
    fn project(&self, input: T) -> Result<Self::Value>;
}

pub trait EnumerableBasis<T>: Basis<T>
where Self::Value: Index<usize, Output = ActivationT>
{
    /// Compute the the i<sup>th</sup> element of the projection for the given
    /// `input`.
    ///
    /// __Note:__ the default implementation computes the full basis to access a
    /// single index. More computational efficient methods should be
    /// implemented where possible.
    fn ith(&self, input: T, index: IndexT) -> Result<ActivationT> {
        check_index!(index < self.dim().into() => {
            self.project(input).map(|b| b[index])
        })
    }
}

pub trait Combinators {
    /// Return a stack of this [`Basis`] over another.
    ///
    /// [`Basis`]: trait.Basis.html
    fn stack<T>(self, other: T) -> Stack<Self, T>
    where Self: Sized {
        Stack::new(self, other)
    }

    /// Return the a stack of this [`Basis`] with a single constant feature
    /// term.
    ///
    /// [`Basis`]: trait.Basis.html
    fn with_bias(self) -> Stack<Self, Bias>
    where Self: Sized {
        self.stack(Bias::unit())
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

///////////////////////////////////////////////////////////////////////////////////////////////////
// Bases:
///////////////////////////////////////////////////////////////////////////////////////////////////
mod fourier;
mod kernelised;
mod polynomial;
mod tile_coding;
mod uniform_grid;

pub use self::fourier::Fourier;
pub use self::kernelised::{KernelBasis, Prototype};
pub use self::polynomial::{Chebyshev, Polynomial};
pub use self::tile_coding::TileCoding;
pub use self::uniform_grid::UniformGrid;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Misc bases:
///////////////////////////////////////////////////////////////////////////////////////////////////
mod closure;
mod constant;
mod normalisation;
mod stack;

pub use self::closure::Closure;
pub use self::constant::{Bias, Fixed};
pub use self::normalisation::{L0Normaliser, L1Normaliser, L2Normaliser, LinfNormaliser};
pub use self::stack::Stack;

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
