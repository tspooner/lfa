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

macro_rules! apply_to_features {
    ($features:expr => $dense:expr; $sparse:expr) => {
        match $features {
            Features::Dense(_) => $dense,
            Features::Sparse(_) => $sparse,
        }
    };

    ($features:expr => $dense:ident, $dbody:block; $sparse:ident, $sbody:block) => {
        match $features {
            Features::Dense($dense) => $dbody,
            Features::Sparse($sparse) => $sbody,
        }
    };
    ($features:expr => mut $dense:ident, $dbody:block; $sparse:ident, $sbody:block) => {
        match $features {
            Features::Dense(ref mut $dense) => $dbody,
            Features::Sparse(ref $sparse) => $sbody,
        }
    };
    ($features:expr => $dense:ident, $dbody:block; mut $sparse:ident, $sbody:block) => {
        match $features {
            Features::Dense(ref $dense) => $dbody,
            Features::Sparse(ref mut $sparse) => $sbody,
        }
    };
    ($features:expr => mut $dense:ident, $dbody:block; mut $sparse:ident, $sbody:block) => {
        match $features {
            Features::Dense(ref mut $dense) => $dbody,
            Features::Sparse(ref mut $sparse) => $sbody,
        }
    };
}

macro_rules! apply_to_dense_or_sparse {
    ($call:ident => $features:expr) => {
        match $features {
            Features::Dense(ref activations) => $call(activations),
            Features::Sparse(ref indices) => $call(indices),
        }
    };
    ($call:ident => $features:expr, $($arg:expr),*) => {
        match $features {
            Features::Dense(ref activations) => $call(activations, $($arg),*),
            Features::Sparse(ref indices) => $call(activations, $($arg),*),
        }
    };
    ($self:ident.$call:ident => $features:expr) => {
        match $features {
            Features::Dense(ref activations) => $self.$call(activations),
            Features::Sparse(ref indices) => $self.$call(indices),
        }
    };
    ($self:ident.$call:ident => $features:expr, $($arg:expr),*) => {
        match $features {
            Features::Dense(ref activations) => $self.$call(activations, $($arg),*),
            Features::Sparse(ref indices) => $self.$call(indices, $($arg),*),
        }
    };
}

/// Projected feature vector representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Features {
    /// Dense, floating-point activation vector.
    Dense(DenseT),

    /// Sparse, index-based activation vector.
    ///
    /// Note: it is taken that all active indices have implied activation of 1.
    Sparse(SparseT),
}

pub use self::Features::{Dense as DenseFeatures, Sparse as SparseFeatures};

impl Features {
    /// Return true if the features is the `Dense` variant.
    pub fn is_dense(&self) -> bool {
        apply_to_features!(self => true; false)
    }

    /// Return true if the features is the `Sparse` variant.
    pub fn is_sparse(&self) -> bool {
        apply_to_features!(self => false; true)
    }

    /// Return the number of active features.
    pub fn activity(&self) -> usize {
        apply_to_features!(self => activations, {
            activations.iter().filter(|v| v.abs() > 1e-7).count()
        }; indices, {
            indices.len()
        })
    }

    /// Remove one feature entry from the features, if present.
    ///
    /// For the `Features::Dense` variant, the feature is set to zero, and for
    /// the `Features::Sparse` variant, the feature index is removed
    /// entirely.
    ///
    /// ```
    /// use lfa::basis::Features;
    ///
    /// let mut dense: Features = vec![0.0, 0.2, 0.4, 0.4].into();
    /// let mut sparse: Features = vec![0, 10, 15].into();
    ///
    /// dense.remove(1);
    /// sparse.remove(10);
    ///
    /// assert_eq!(dense, vec![0.0, 0.0, 0.4, 0.4].into());
    /// assert_eq!(sparse, vec![0, 15].into());
    /// ```
    pub fn remove(&mut self, idx: usize) {
        apply_to_features!(self => mut activations, {
            if let Some(a) = activations.get_mut(idx) {
                *a = 0.0;
            }
        }; mut indices, {
            indices.remove(&idx);
        })
    }

    /// Apply the dot product operation between the `Features` and some other
    /// `Vector`, typically a set of weights.
    ///
    /// ```
    /// use lfa::basis::Features;
    /// use lfa::geometry::Vector;
    ///
    /// let weights = Vector::from_vec(vec![2.0, 5.0, 1.0]);
    ///
    /// assert_eq!(Features::dot(&vec![0.0, 0.2, 0.8].into(), &weights.view()), 1.8);
    /// assert_eq!(Features::dot(&vec![0, 1].into(), &weights.view()), 7.0);
    /// ```
    pub fn dot(&self, weights: &VectorView<f64>) -> f64 {
        apply_to_features!(self => activations, {
            Features::dot_dense(activations, weights)
        }; indices, {
            Features::dot_sparse(indices, weights)
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

    /// Apply the dot product operation between the `Features` and some other
    /// `Vector`, typically a set of weights.
    ///
    /// ```
    /// use lfa::basis::Features;
    /// use lfa::geometry::{Matrix, Vector};
    ///
    /// let weights = Matrix::from_shape_vec((3, 2), vec![2.0, 5.0, 1.0, 3.0, 1.0, 3.0]).unwrap();
    ///
    /// assert!(
    ///     Features::matmul(&vec![0.1, 0.2, 0.7].into(), &weights.view()).all_close(
    ///         &Vector::from_vec(vec![1.1, 3.2]),
    ///         1e-7 // eps
    ///     )
    /// );
    /// assert_eq!(
    ///     Features::matmul(&vec![0, 1].into(), &weights.view()),
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

    /// Expand the features and convert it into a raw, dense vector.
    ///
    /// ```
    /// use lfa::basis::Features;
    ///
    /// assert_eq!(
    ///     Features::expanded(vec![0, 2, 1, 4].into(), 5),
    ///     vec![1.0, 1.0, 1.0, 0.0, 1.0].into()
    /// );
    /// ```
    pub fn expanded(self, dim: usize) -> DenseT {
        apply_to_features!(self => activations, {
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

    /// Apply the function `f` to the features if the `Dense` variant or
    /// return `None`.
    pub fn map_dense<F, T>(self, f: impl FnOnce(DenseT) -> T) -> Option<T> {
        apply_to_features!(self => activations, {
            Some(f(activations))
        }; indices, {
            None
        })
    }

    /// Apply the function `f` to the features if the `Sparse` variant or
    /// return `None`.
    pub fn map_sparse<F, T>(self, f: impl FnOnce(SparseT) -> T) -> Option<T> {
        apply_to_features!(self => activations, {
            None
        }; indices, {
            Some(f(indices))
        })
    }

    /// Apply the function `f` or `g` depending on the contents of the features; either `Dense` or
    /// `Sparse`, respectively.
    pub fn map_either<T>(
        self,
        f_dense: impl FnOnce(DenseT) -> T,
        f_sparse: impl FnOnce(SparseT) -> T,
    ) -> T
    {
        apply_to_features!(self => activations, {
            f_dense(activations)
        }; indices, {
            f_sparse(indices)
        })
    }
}

impl Add<Features> for Features {
    type Output = Features;

    fn add(self, rhs: Features) -> Features {
        match (self, rhs) {
            (SparseFeatures(idx1), SparseFeatures(idx2)) => {
                SparseFeatures(idx1.union(&idx2).cloned().collect())
            },
            (DenseFeatures(act1), DenseFeatures(act2)) => {
                DenseFeatures(stack(Axis(0), &[act1.view(), act2.view()]).unwrap())
            },
            _ => unimplemented!(
                "Cannot combine dense/sparse with no knowledge of the full \
                 dimensionality of sparse features."
            ),
        }
    }
}

impl Index<usize> for Features {
    type Output = f64;

    fn index(&self, idx: usize) -> &f64 {
        apply_to_features!(self => activations, {
            activations.index(idx)
        }; indices, {
            if indices.contains(&idx) {
                &1.0
            } else {
                &0.0
            }
        })
    }
}

impl PartialEq<Features> for Features {
    fn eq(&self, rhs: &Features) -> bool {
        match (self, rhs) {
            (&SparseFeatures(ref idx1), &SparseFeatures(ref idx2)) => idx1.eq(&idx2),
            (&DenseFeatures(ref act1), &DenseFeatures(ref act2)) => act1.eq(&act2),
            _ => unimplemented!(
                "Cannot check equality of dense/sparse with no knowledge of the \
                 full dimensionality of sparse features."
            ),
        }
    }
}

impl From<DenseT> for Features {
    fn from(activations: DenseT) -> Features { DenseFeatures(activations) }
}

impl From<Vec<ActivationT>> for Features {
    fn from(activations: Vec<ActivationT>) -> Features {
        DenseFeatures(Vector::from_vec(activations))
    }
}

impl FromIterator<ActivationT> for Features {
    fn from_iter<I: IntoIterator<Item = ActivationT>>(iter: I) -> Self {
        DenseFeatures(Vector::from_iter(iter))
    }
}

impl From<SparseT> for Features {
    fn from(indices: SparseT) -> Features { SparseFeatures(indices) }
}

impl From<Vec<IndexT>> for Features {
    fn from(indices: Vec<IndexT>) -> Features { Features::from_iter(indices.into_iter()) }
}

impl FromIterator<IndexT> for Features {
    fn from_iter<I: IntoIterator<Item = IndexT>>(iter: I) -> Self {
        SparseFeatures({
            let mut is = IndexSet::new();

            for v in iter {
                is.insert(v);
            }

            is
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Features, Vector};

    #[test]
    fn test_sparse() {
        let mut f = Features::from(vec![0usize, 5usize, 10usize]);

        assert!(f.is_sparse());
        assert!(!f.is_dense());
        assert_eq!(f.activity(), 3);

        f.remove(1);
        assert_eq!(f, Features::from(vec![0usize, 5usize, 10usize]));

        assert_eq!(f.dot(&Vector::ones(11).view()), 3.0);
        assert_eq!(f.dot(&Vector::zeros(11).view()), 0.0);

        f.remove(5);
        assert_eq!(f, Features::from(vec![0usize, 10usize]));

        assert_eq!(f.dot(&Vector::ones(11).view()), 2.0);
        assert_eq!(f.dot(&Vector::zeros(11).view()), 0.0);

        assert_eq!(f[0], 1.0);
        assert_eq!(f[1], 0.0);
        assert_eq!(f[5], 0.0);
        assert_eq!(f[10], 1.0);
    }

    #[test]
    fn test_dense() {
        let mut f = Features::from(vec![0.0, 0.1, 0.2, 0.1, 0.0, 0.6]);

        assert!(f.is_dense());
        assert!(!f.is_sparse());
        assert_eq!(f.activity(), 4);

        f.remove(10);
        assert_eq!(f, Features::from(vec![0.0, 0.1, 0.2, 0.1, 0.0, 0.6]));

        assert_eq!(f.dot(&Vector::ones(6).view()), 1.0);
        assert_eq!(f.dot(&Vector::zeros(6).view()), 0.0);

        f.remove(1);
        assert_eq!(f, Features::from(vec![0.0, 0.0, 0.2, 0.1, 0.0, 0.6]));

        assert_eq!(f.dot(&Vector::ones(6).view()), 0.9);
        assert_eq!(f.dot(&Vector::zeros(6).view()), 0.0);

        assert_eq!(f[0], 0.0);
        assert_eq!(f[1], 0.0);
        assert_eq!(f[2], 0.2);
        assert_eq!(f[3], 0.1);
        assert_eq!(f[4], 0.0);
        assert_eq!(f[5], 0.6);
    }
}
