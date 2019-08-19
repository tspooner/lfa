//! Module for basis function representations used by `Approximator` types.
use crate::{IndexT, ActivationT, Features};

pub mod kernels;

pub struct ProjectorIterator<'a, P>(&'a P);

/// Trait for basis projectors.
pub trait Projector {
    /// Return the number of features produced by this `Projector`.
    fn n_features(&self) -> usize;

    fn project_ith(&self, input: &[f64], index: IndexT) -> Option<ActivationT>;

    /// Project data from an input space onto the basis.
    ///
    /// ```
    /// use lfa::{Features, basis::{Projector, Constant}};
    ///
    /// let projector = Constant::unit();
    ///
    /// assert_eq!(
    ///     projector.project(&[0.0]),
    ///     Features::from(vec![1.0])
    /// );
    /// ```
    fn project(&self, input: &[f64]) -> Features {
        (0..self.n_features())
            .into_iter()
            .map(|i| self.project_ith(input, i).unwrap_or(0.0))
            .collect()
    }

    /// Return a `Stack` of this `Projector` over another.
    fn stack<P: Projector>(self, other: P) -> Stacker<Self, P> where Self: Sized {
        Stacker::new(self, other)
    }

    /// Return the a `Stack` of this `Projector` with a single constant feature term.
    fn with_constant(self) -> Stacker<Self, Constant> where Self: Sized {
        self.stack(Constant::unit())
    }

    fn normalise_l0(self) -> L0Normaliser<Self> where Self: Sized {
        L0Normaliser::new(self)
    }

    /// Return the original `Projector` with all activations normalised in _L₁_.
    fn normalise_l1(self) -> L1Normaliser<Self> where Self: Sized {
        L1Normaliser::new(self)
    }

    /// Return the original `Projector` with all activations normalised in _L₂_.
    fn normalise_l2(self) -> L2Normaliser<Self> where Self: Sized {
        L2Normaliser::new(self)
    }

    fn normalise_linf(self) -> LinfNormaliser<Self> where Self: Sized {
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
