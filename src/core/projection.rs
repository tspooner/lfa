#![macro_use]
use crate::{
    core::*,
    geometry::{MatrixView, Vector, VectorView},
};
use ndarray::{stack, Axis};
use std::{
    iter::FromIterator,
    ops::{Add, Index},
};

macro_rules! apply_to_projection {
    ($projection:expr => $dense:expr; $sparse:expr) => {
        match $projection {
            Projection::Dense(_) => $dense,
            Projection::Sparse(_) => $sparse,
        }
    };

    ($projection:expr => $dense:ident, $dbody:block; $sparse:ident, $sbody:block) => {
        match $projection {
            Projection::Dense($dense) => $dbody,
            Projection::Sparse($sparse) => $sbody,
        }
    };
    ($projection:expr => mut $dense:ident, $dbody:block; $sparse:ident, $sbody:block) => {
        match $projection {
            Projection::Dense(ref mut $dense) => $dbody,
            Projection::Sparse(ref $sparse) => $sbody,
        }
    };
    ($projection:expr => $dense:ident, $dbody:block; mut $sparse:ident, $sbody:block) => {
        match $projection {
            Projection::Dense(ref $dense) => $dbody,
            Projection::Sparse(ref mut $sparse) => $sbody,
        }
    };
    ($projection:expr => mut $dense:ident, $dbody:block; mut $sparse:ident, $sbody:block) => {
        match $projection {
            Projection::Dense(ref mut $dense) => $dbody,
            Projection::Sparse(ref mut $sparse) => $sbody,
        }
    };
}

macro_rules! apply_to_dense_or_sparse {
    ($call:ident => $projection:expr) => {
        match $projection {
            Projection::Dense(ref activations) => $call(activations),
            Projection::Sparse(ref indices) => $call(indices),
        }
    };
    ($call:ident => $projection:expr, $($arg:expr),*) => {
        match $projection {
            Projection::Dense(ref activations) => $call(activations, $($arg),*),
            Projection::Sparse(ref indices) => $call(activations, $($arg),*),
        }
    };
    ($self:ident.$call:ident => $projection:expr) => {
        match $projection {
            Projection::Dense(ref activations) => $self.$call(activations),
            Projection::Sparse(ref indices) => $self.$call(indices),
        }
    };
    ($self:ident.$call:ident => $projection:expr, $($arg:expr),*) => {
        match $projection {
            Projection::Dense(ref activations) => $self.$call(activations, $($arg),*),
            Projection::Sparse(ref indices) => $self.$call(indices, $($arg),*),
        }
    };
}

/// Projected feature vector representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Projection {
    /// Dense, floating-point activation vector.
    Dense(DenseT),

    /// Sparse, index-based activation vector.
    ///
    /// Note: it is taken that all active indices have implied activation of 1.
    Sparse(SparseT),
}

pub use self::Projection::{Dense as DenseProjection, Sparse as SparseProjection};

impl Projection {
    /// Return true if the projection is the `Dense` variant.
    pub fn is_dense(&self) -> bool {
        apply_to_projection!(self => true; false)
    }

    /// Return true if the projection is the `Sparse` variant.
    pub fn is_sparse(&self) -> bool {
        apply_to_projection!(self => false; true)
    }

    /// Return the number of active features in the projection.
    pub fn activity(&self) -> usize {
        apply_to_projection!(self => activations, {
            activations.len()
        }; indices, {
            indices.iter().max().cloned().unwrap_or(0)
        })
    }

    /// Remove one feature entry from the projection, if present.
    ///
    /// For the `Projection::Dense` variant, the feature is set to zero, and for
    /// the `Projection::Sparse` variant, the feature index is removed
    /// entirely.
    ///
    /// ```
    /// use lfa::basis::Projection;
    ///
    /// let mut dense: Projection = vec![0.0, 0.2, 0.4, 0.4].into();
    /// let mut sparse: Projection = vec![0, 10, 15].into();
    ///
    /// dense.remove(1);
    /// sparse.remove(10);
    ///
    /// assert_eq!(dense, vec![0.0, 0.0, 0.4, 0.4].into());
    /// assert_eq!(sparse, vec![0, 15].into());
    /// ```
    pub fn remove(&mut self, idx: usize) {
        apply_to_projection!(self => mut activations, {
            activations[idx] = 0.0;
        }; mut indices, {
            indices.remove(&idx);
        })
    }

    /// Apply the dot product operation between the `Projection` and some other
    /// `Vector`, typically a set of weights.
    ///
    /// ```
    /// use lfa::basis::Projection;
    /// use lfa::geometry::Vector;
    ///
    /// let weights = Vector::from_vec(vec![2.0, 5.0, 1.0]);
    ///
    /// assert_eq!(Projection::dot(&vec![0.0, 0.2, 0.8].into(), &weights.view()), 1.8);
    /// assert_eq!(Projection::dot(&vec![0, 1].into(), &weights.view()), 7.0);
    /// ```
    pub fn dot(&self, weights: &VectorView<f64>) -> f64 {
        apply_to_projection!(self => activations, {
            Projection::dot_dense(activations, weights)
        }; indices, {
            Projection::dot_sparse(indices, weights)
        })
    }

    pub(crate) fn dot_dense(activations: &DenseT, weights: &VectorView<f64>) -> f64 {
        activations.dot(weights)
    }

    pub(crate) fn dot_sparse(indices: &SparseT, weights: &VectorView<f64>) -> f64 {
        indices
            .iter()
            .fold(0.0, |acc, idx| acc + weights[*idx])
    }

    /// Apply the dot product operation between the `Projection` and some other
    /// `Vector`, typically a set of weights.
    ///
    /// ```
    /// use lfa::basis::Projection;
    /// use lfa::geometry::{Matrix, Vector};
    ///
    /// let weights = Matrix::from_shape_vec((3, 2), vec![2.0, 5.0, 1.0, 3.0, 1.0, 3.0]).unwrap();
    ///
    /// assert!(
    ///     Projection::matmul(&vec![0.1, 0.2, 0.7].into(), &weights.view()).all_close(
    ///         &Vector::from_vec(vec![1.1, 3.2]),
    ///         1e-7 // eps
    ///     )
    /// );
    /// assert_eq!(
    ///     Projection::matmul(&vec![0, 1].into(), &weights.view()),
    ///     Vector::from_vec(vec![3.0, 8.0])
    /// );
    /// ```
    pub fn matmul(&self, weights: &MatrixView<f64>) -> Vector<f64> {
        weights
            .gencolumns()
            .into_iter()
            .map(|col| self.dot(&col))
            .collect()
    }

    /// Expand the projection and convert it into a raw, dense vector.
    ///
    /// ```
    /// use lfa::basis::Projection;
    ///
    /// assert_eq!(
    ///     Projection::expanded(vec![0, 2, 1, 4].into(), 5),
    ///     vec![1.0, 1.0, 1.0, 0.0, 1.0].into()
    /// );
    /// ```
    pub fn expanded(self, dim: usize) -> DenseT {
        apply_to_projection!(self => activations, {
            if activations.len() != dim {
                let mut activations = activations.to_vec();
                activations.resize(dim, 0.0);

                DenseT::from_vec(activations)
            } else {
                activations.to_owned()
            }
        }; indices, {
            let mut phi = Vector::zeros((dim,));
            for idx in indices.iter() {
                phi[*idx] = 1.0;
            }

            phi
        })
    }

    /// Apply the function `f` to the projection if the `Dense` variant or
    /// return `None`.
    pub fn map_dense<F, T>(self, f: impl FnOnce(DenseT) -> T) -> Option<T> {
        apply_to_projection!(self => activations, {
            Some(f(activations))
        }; indices, {
            None
        })
    }

    /// Apply the function `f` to the projection if the `Sparse` variant or
    /// return `None`.
    pub fn map_sparse<F, T>(self, f: impl FnOnce(SparseT) -> T) -> Option<T> {
        apply_to_projection!(self => activations, {
            None
        }; indices, {
            Some(f(indices))
        })
    }

    /// Apply the function `f` or `g` depending on the contents of the
    /// projection; either `Dense` or `Sparse`, respectively.
    pub fn map_either<T>(
        self,
        f_dense: impl FnOnce(DenseT) -> T,
        f_sparse: impl FnOnce(SparseT) -> T,
    ) -> T
    {
        apply_to_projection!(self => activations, {
            f_dense(activations)
        }; indices, {
            f_sparse(indices)
        })
    }
}

impl Add<Projection> for Projection {
    type Output = Projection;

    fn add(self, rhs: Projection) -> Projection {
        match (self, rhs) {
            (SparseProjection(idx1), SparseProjection(idx2)) => {
                SparseProjection(idx1.union(&idx2).cloned().collect())
            },
            (DenseProjection(act1), DenseProjection(act2)) => {
                DenseProjection(stack(Axis(0), &[act1.view(), act2.view()]).unwrap())
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
        apply_to_projection!(self => activations, {
            activations.index(idx)
        }; indices, {
            if idx < indices.len() {
                &1.0
            } else {
                &0.0
            }
        })
    }
}

impl PartialEq<Projection> for Projection {
    fn eq(&self, rhs: &Projection) -> bool {
        match (self, rhs) {
            (&SparseProjection(ref idx1), &SparseProjection(ref idx2)) => idx1.eq(&idx2),
            (&DenseProjection(ref act1), &DenseProjection(ref act2)) => act1.eq(&act2),
            _ => unimplemented!(
                "Cannot check equality of dense/sparse with no knowledge of the \
                 full dimensionality of sparse projection."
            ),
        }
    }
}

impl From<DenseT> for Projection {
    fn from(activations: DenseT) -> Projection { DenseProjection(activations) }
}

impl From<Vec<ActivationT>> for Projection {
    fn from(activations: Vec<ActivationT>) -> Projection {
        DenseProjection(Vector::from_vec(activations))
    }
}

impl FromIterator<ActivationT> for Projection {
    fn from_iter<I: IntoIterator<Item = ActivationT>>(iter: I) -> Self {
        DenseProjection(Vector::from_iter(iter))
    }
}

impl From<SparseT> for Projection {
    fn from(indices: SparseT) -> Projection { SparseProjection(indices) }
}

impl From<Vec<IndexT>> for Projection {
    fn from(indices: Vec<IndexT>) -> Projection { Projection::from_iter(indices.into_iter()) }
}

impl FromIterator<IndexT> for Projection {
    fn from_iter<I: IntoIterator<Item = IndexT>>(iter: I) -> Self {
        SparseProjection({
            let mut is = IndexSet::new();

            for v in iter {
                is.insert(v);
            }

            is
        })
    }
}
