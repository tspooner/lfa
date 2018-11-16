use super::{ActivationT, DenseT, IndexSet, IndexT, SparseT};
use geometry::Vector;
use ndarray::{stack, Axis};
use std::iter::FromIterator;
use std::ops::{Add, Index};

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
        use Projection::*;

        match self {
            &mut Dense(ref mut activations) => activations[idx] = 0.0,
            &mut Sparse(ref mut active_indices) => {
                active_indices.remove(&idx);
            },
        }
    }

    /// Return the maximum number of active features in the basis space.
    pub fn activity(&self) -> usize {
        use Projection::*;

        match self {
            &Dense(ref activations) => activations.len(),
            &Sparse(ref active_indices) => active_indices.len(),
        }
    }

    /// Expand and normalise a given projection, and convert into a raw, dense
    /// vector.
    pub fn expanded(self, dim: usize) -> DenseT {
        use Projection::*;

        #[inline]
        fn expand_dense(phi: DenseT, size: usize) -> DenseT {
            let mut phi = phi.to_vec();
            phi.resize(size, 0.0);

            DenseT::from_vec(phi)
        }

        #[inline]
        fn expand_sparse(active_indices: SparseT, size: usize) -> DenseT {
            let mut phi = Vector::zeros((size,));
            let activation = 1.0 / active_indices.len() as f64;

            for idx in active_indices.iter() {
                phi[*idx] = activation;
            }

            phi
        }

        match self {
            Dense(phi) => expand_dense(phi, dim),
            Sparse(active_indices) => expand_sparse(active_indices, dim),
        }
    }
}

impl Add<Projection> for Projection {
    type Output = Projection;

    fn add(self, rhs: Projection) -> Projection {
        use Projection::*;

        match (self, rhs) {
            (Sparse(idx1), Sparse(idx2)) => Sparse(idx1.union(&idx2).cloned().collect()),
            (Dense(act1), Dense(act2)) => {
                Dense(stack(Axis(0), &[act1.view(), act2.view()]).unwrap())
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
        use Projection::*;

        match (self, rhs) {
            (&Sparse(ref idx1), &Sparse(ref idx2)) => idx1.eq(&idx2),
            (&Dense(ref act1), &Dense(ref act2)) => act1.eq(&act2),

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
