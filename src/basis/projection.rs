use crate::core::*;
use crate::geometry::{Vector, Matrix, norms::{l1, l2}};
use crate::core::IndexSet;
use ndarray::{stack, Axis};
use std::{
    iter::FromIterator,
    ops::{Add, Index},
};

/// Projected feature vector representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Projection {
    /// Dense, floating-point activation vector.
    Dense(DenseT),

    /// Sparse, index-based activation vector.
    Sparse(SparseT),
}

pub use self::Projection::{
    Dense as DenseProjection,
    Sparse as SparseProjection,
};

impl Projection {
    /// Return true if the projection is the `Dense` variant.
    pub fn is_dense(&self) -> bool {
        match *self {
            DenseProjection(_) => true,
            SparseProjection(_) => false,
        }
    }

    /// Return true if the projection is the `Sparse` variant.
    pub fn is_sparse(&self) -> bool {
        match *self {
            DenseProjection(_) => false,
            SparseProjection(_) => true,
        }
    }

    /// Return the number of active features in the projection.
    pub fn activity(&self) -> usize {
        match self {
            &DenseProjection(ref activations) => activations.len(),
            &SparseProjection(ref active_indices) =>
                active_indices.iter().max().cloned().unwrap_or(0),
        }
    }

    /// Remove one feature entry from the projection, if present.
    ///
    /// For the `Projection::Dense` variant, the feature is set to zero, and for the
    /// `Projection::Sparse` variant, the feature index is removed entirely.
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
        match *self {
            DenseProjection(ref mut activations) => activations[idx] = 0.0,
            SparseProjection(ref mut active_indices) => { active_indices.remove(&idx); },
        }
    }

    /// Apply the dot product operation between the `Projection` and some other `Vector`, typically
    /// a set of weights.
    ///
    /// ```
    /// use lfa::basis::Projection;
    /// use lfa::geometry::Vector;
    ///
    /// let weights = Vector::from_vec(vec![2.0, 5.0, 1.0]);
    ///
    /// assert_eq!(Projection::dot(&vec![0.0, 0.2, 0.8].into(), &weights), 1.8);
    /// assert_eq!(Projection::dot(&vec![0, 1].into(), &weights), 7.0);
    /// ```
    pub fn dot(&self, weights: &Vector<f64>) -> f64 {
        match self {
            DenseProjection(ref activations) => activations.dot(weights),
            SparseProjection(ref indices) =>
                indices.iter().rev().fold(0.0, |acc, idx| acc + weights[*idx]),
        }
    }

    /// Apply the dot product operation between the `Projection` and some other `Vector`, typically
    /// a set of weights.
    ///
    /// ```
    /// use lfa::basis::Projection;
    /// use lfa::geometry::{Vector, Matrix};
    ///
    /// let weights = Matrix::from_shape_vec((3, 2), vec![
    ///     2.0, 5.0,
    ///     1.0, 3.0,
    ///     1.0, 3.0,
    /// ]).unwrap();
    ///
    /// assert!(Projection::matmul(
    ///     &vec![0.1, 0.2, 0.7].into(), &weights
    /// ).all_close(
    ///     &Vector::from_vec(vec![1.1, 3.2]), 1e-7 // eps
    /// ));
    /// assert_eq!(
    ///     Projection::matmul(&vec![0, 1].into(), &weights),
    ///     Vector::from_vec(vec![3.0, 8.0])
    /// );
    /// ```
    pub fn matmul(&self, weights: &Matrix<f64>) -> Vector<f64> {
        match self {
            DenseProjection(ref activations) =>
                activations.view().insert_axis(Axis(0)).dot(weights).index_axis_move(Axis(0), 0),
            SparseProjection(ref indices) => (0..weights.cols()).map(|col| {
                indices.iter().rev().fold(0.0, |acc, idx| acc + weights[(*idx, col)])
            }).collect(),
        }
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
        match self {
            DenseProjection(phi) => {
                let mut phi = phi.to_vec();
                phi.resize(dim, 0.0);

                DenseT::from_vec(phi)
            },
            SparseProjection(active_indices) => {
                let mut phi = Vector::zeros((dim,));
                for idx in active_indices.iter() {
                    phi[*idx] = 1.0;
                }

                phi
            },
        }
    }

    /// Return an expanded feature vector with L<sub>1</sub> normalisation applied.
    ///
    /// ```
    /// use lfa::basis::Projection;
    ///
    /// assert_eq!(
    ///     Projection::expanded_l1(vec![0, 2, 1, 4].into(), 5),
    ///     vec![0.25, 0.25, 0.25, 0.0, 0.25].into()
    /// );
    /// ```
    pub fn expanded_l1(self, dim: usize) -> DenseT {
        let phi = self.expanded(dim);
        let z = l1(phi.as_slice().unwrap());

        phi / z
    }

    /// Return an expanded feature vector with L<sub>2</sub> normalisation applied.
    ///
    /// ```
    /// use lfa::basis::Projection;
    ///
    /// assert_eq!(
    ///     Projection::expanded_l2(vec![0, 2, 1, 4].into(), 5),
    ///     vec![0.5, 0.5, 0.5, 0.0, 0.5].into()
    /// );
    /// ```
    pub fn expanded_l2(self, dim: usize) -> DenseT {
        let phi = self.expanded(dim);
        let z = l2(phi.as_slice().unwrap());

        phi / z
    }

    /// Apply the function `f` to the projection if the `Dense` variant or return `None`.
    pub fn map_dense<F, T>(self, f: impl FnOnce(DenseT) -> T) -> Option<T> {
        match self {
            DenseProjection(activations) => Some(f(activations)),
            SparseProjection(_) => None,
        }
    }

    /// Apply the function `f` to the projection if the `Sparse` variant or return `None`.
    pub fn map_sparse<F, T>(self, f: impl FnOnce(SparseT) -> T) -> Option<T> {
        match self {
            DenseProjection(_) => None,
            SparseProjection(indices) => Some(f(indices)),
        }
    }

    /// Apply the function `f` or `g` depending on the contents of the projection; either `Dense`
    /// or `Sparse`, respectively.
    pub fn map_either<T>(
        self,
        f_dense: impl FnOnce(DenseT) -> T,
        f_sparse: impl FnOnce(SparseT) -> T
    ) -> T
    {
        match self {
            DenseProjection(activations) => f_dense(activations),
            SparseProjection(indices) => f_sparse(indices),
        }
    }
}

impl Add<Projection> for Projection {
    type Output = Projection;

    fn add(self, rhs: Projection) -> Projection {
        match (self, rhs) {
            (SparseProjection(idx1), SparseProjection(idx2)) =>
                SparseProjection(idx1.union(&idx2).cloned().collect()),
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
        match *self {
            DenseProjection(ref activations) => activations.index(idx),
            SparseProjection(ref active_indices) => {
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
            (&SparseProjection(ref idx1), &SparseProjection(ref idx2)) => idx1.eq(&idx2),
            (&DenseProjection(ref act1), &DenseProjection(ref act2)) => act1.eq(&act2),
            _ => unimplemented!(
                "Cannot check equality of dense/sparse with no knowledge of the \
                 full dimensionality of sparse projection."
            ),
        }
    }
}

impl Into<Projection> for DenseT {
    fn into(self) -> Projection { DenseProjection(self) }
}

impl Into<Projection> for Vec<ActivationT> {
    fn into(self) -> Projection { DenseProjection(Vector::from_vec(self)) }
}

impl FromIterator<ActivationT> for Projection {
    fn from_iter<I: IntoIterator<Item = ActivationT>>(iter: I) -> Self {
        DenseProjection(Vector::from_iter(iter))
    }
}

impl Into<Projection> for SparseT {
    fn into(self) -> Projection { SparseProjection(self) }
}

impl Into<Projection> for Vec<IndexT> {
    fn into(self) -> Projection { Projection::from_iter(self.into_iter()) }
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
