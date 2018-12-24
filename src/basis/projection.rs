use crate::basis::CandidateFeature;
use crate::core::*;
use crate::geometry::{Space, Vector, norms::{l1, l2, linf}};
use crate::core::IndexSet;
use ndarray::{stack, Axis};
use std::{
    collections::HashMap,
    iter::FromIterator,
    ops::{Add, Index},
};

/// Trait for basis projectors.
pub trait Projector<I: ?Sized>: Space<Value = Projection> {
    /// Project data from an input space onto the basis.
    fn project(&self, input: &I) -> Projection;

    /// Project data from an input space onto the basis and convert into a raw,
    /// dense vector.
    fn project_expanded(&self, input: &I) -> DenseT { self.project(input).expanded(self.dim()) }
}

/// Trait for projectors with adaptive bases.
pub trait AdaptiveProjector<I: ?Sized>: Projector<I> {
    fn discover(&mut self, input: &I, error: f64) -> Option<HashMap<usize, IndexSet>>;
    fn add_feature(&mut self, feature: CandidateFeature) -> Option<(usize, IndexSet)>;
}

impl<P: Projector<[f64]>> Projector<Vec<f64>> for P {
    fn project(&self, input: &Vec<f64>) -> Projection {
        Projector::<[f64]>::project(self, &input)
    }
}

impl<P: Projector<[f64]>> Projector<Vector<f64>> for P {
    fn project(&self, input: &Vector<f64>) -> Projection {
        Projector::<[f64]>::project(self, input.as_slice().unwrap())
    }
}

macro_rules! impl_fixed {
    ($($n:expr),*) => {
        $(
            impl<P: Projector<[f64]>> Projector<[f64; $n]> for P {
                fn project(&self, input: &[f64; $n]) -> Projection {
                    Projector::<[f64]>::project(self, input)
                }
            }
        )*
    }
}

impl_fixed!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24);

/// Projected feature vector representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Projection {
    /// Dense, floating-point activation vector.
    Dense(DenseT),

    /// Sparse, index-based activation vector.
    Sparse(SparseT),
}

impl Projection {
    /// Remove/set to zero one feature entry of the projection.
    pub fn remove(&mut self, idx: usize) {
        match self {
            &mut Projection::Dense(ref mut activations) => activations[idx] = 0.0,
            &mut Projection::Sparse(ref mut active_indices) => {
                active_indices.remove(&idx);
            },
        }
    }

    /// Return the maximum number of active features in the basis space.
    pub fn activity(&self) -> usize {
        match self {
            &Projection::Dense(ref activations) => activations.len(),
            &Projection::Sparse(ref active_indices) => active_indices.len(),
        }
    }

    /// Expand the projection and convert it into a raw, dense vector.
    pub fn expanded(self, dim: usize) -> DenseT {
        #[inline]
        fn expand_dense(phi: DenseT, size: usize) -> DenseT {
            let mut phi = phi.to_vec();
            phi.resize(size, 0.0);

            DenseT::from_vec(phi)
        }

        #[inline]
        fn expand_sparse(active_indices: SparseT, size: usize) -> DenseT {
            let mut phi = Vector::zeros((size,));

            for idx in active_indices.iter() {
                phi[*idx] = 1.0;
            }

            phi
        }

        match self {
            Projection::Dense(phi) => expand_dense(phi, dim),
            Projection::Sparse(active_indices) => expand_sparse(active_indices, dim),
        }
    }

    /// Return an expanded feature vector with L1 normalisation applied.
    pub fn expanded_l1(self, dim: usize) -> DenseT {
        let phi = self.expanded(dim);
        let z = l1(phi.as_slice().unwrap());

        phi / z
    }

    /// Return an expanded feature vector with L2 normalisation applied.
    pub fn expanded_l2(self, dim: usize) -> DenseT {
        let phi = self.expanded(dim);
        let z = l2(phi.as_slice().unwrap());

        phi / z
    }

    /// Return an expanded feature vector with Linf normalisation applied.
    pub fn expanded_linf(self, dim: usize) -> DenseT {
        let phi = self.expanded(dim);
        let z = linf(phi.as_slice().unwrap());

        phi / z
    }
}

impl Add<Projection> for Projection {
    type Output = Projection;

    fn add(self, rhs: Projection) -> Projection {
        match (self, rhs) {
            (Projection::Sparse(idx1), Projection::Sparse(idx2)) =>
                Projection::Sparse(idx1.union(&idx2).cloned().collect()),
            (Projection::Dense(act1), Projection::Dense(act2)) => {
                Projection::Dense(stack(Axis(0), &[act1.view(), act2.view()]).unwrap())
            },

            _ => unimplemented!(
                "Cannot combine dense/sparse with no knowledge of the full \
                 dimensionality of sparse projection."
            ),
        }
    }
}

impl Index<usize> for Projection {
    type Output = f64;

    fn index(&self, idx: usize) -> &f64 {
        match self {
            &Projection::Dense(ref activations) => activations.index(idx),
            &Projection::Sparse(ref active_indices) => {
                if idx < active_indices.len() {
                    &1.0
                } else {
                    &0.0
                }
            },
        }
    }
}

impl PartialEq<Projection> for Projection {
    fn eq(&self, rhs: &Projection) -> bool {
        match (self, rhs) {
            (&Projection::Sparse(ref idx1), &Projection::Sparse(ref idx2)) => idx1.eq(&idx2),
            (&Projection::Dense(ref act1), &Projection::Dense(ref act2)) => act1.eq(&act2),

            _ => unimplemented!(
                "Cannot check equality of dense/sparse with no knowledge of the \
                 full dimensionality of sparse projection."
            ),
        }
    }
}

impl Into<Projection> for DenseT {
    fn into(self) -> Projection { Projection::Dense(self) }
}

impl Into<Projection> for Vec<ActivationT> {
    fn into(self) -> Projection { Projection::Dense(Vector::from_vec(self)) }
}

impl FromIterator<ActivationT> for Projection {
    fn from_iter<I: IntoIterator<Item = ActivationT>>(iter: I) -> Self {
        Projection::Dense(Vector::from_iter(iter))
    }
}

impl Into<Projection> for SparseT {
    fn into(self) -> Projection { Projection::Sparse(self) }
}

impl Into<Projection> for Vec<IndexT> {
    fn into(self) -> Projection { Projection::from_iter(self.into_iter()) }
}

impl FromIterator<IndexT> for Projection {
    fn from_iter<I: IntoIterator<Item = IndexT>>(iter: I) -> Self {
        Projection::Sparse({
            let mut is = IndexSet::new();

            for v in iter {
                is.insert(v);
            }

            is
        })
    }
}