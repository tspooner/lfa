#![macro_use]
use ndarray::{Array1, ArrayView1, ArrayViewMut1, ArrayView2};
use std::{iter::FromIterator, ops::{AddAssign, Index}};

pub type IndexT = usize;
pub type ActivationT = f64;

pub type DenseT = Array1<ActivationT>;
pub type SparseT = ::std::collections::HashMap<IndexT, ActivationT>;

macro_rules! apply_to_features {
    ($features:expr => $dense:expr; $sparse:expr) => {
        match $features {
            Features::Dense(_) => $dense,
            Features::Sparse(_, _) => $sparse,
        }
    };
    ($features:expr => $dense:ident, $dbody:block; $sparse:ident, $sbody:block) => {
        match $features {
            Features::Dense($dense) => $dbody,
            Features::Sparse(_, $sparse) => $sbody,
        }
    };
    ($features:expr => ref $dense:ident, $dbody:block; ref $sparse:ident, $sbody:block) => {
        match $features {
            Features::Dense(ref $dense) => $dbody,
            Features::Sparse(_, ref $sparse) => $sbody,
        }
    };
    ($features:expr => mut $dense:ident, $dbody:block; $sparse:ident, $sbody:block) => {
        match $features {
            Features::Dense(ref mut $dense) => $dbody,
            Features::Sparse(_, ref $sparse) => $sbody,
        }
    };
    ($features:expr => $dense:ident, $dbody:block; mut $sparse:ident, $sbody:block) => {
        match $features {
            Features::Dense(ref $dense) => $dbody,
            Features::Sparse(_, ref mut $sparse) => $sbody,
        }
    };
    ($features:expr => mut $dense:ident, $dbody:block; mut $sparse:ident, $sbody:block) => {
        match $features {
            Features::Dense(ref mut $dense) => $dbody,
            Features::Sparse(_, ref mut $sparse) => $sbody,
        }
    };
}

#[allow(unused_macros)]
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
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum Features {
    /// Dense, floating-point activation vector.
    Dense(DenseT),

    /// Sparse, index-based activation vector.
    Sparse(usize, SparseT),
}

pub(crate) use self::Features::{Dense as DenseFeatures, Sparse as SparseFeatures};

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
    pub fn n_features(&self) -> usize {
        match self {
            Features::Dense(ref activations) => activations.len(),
            Features::Sparse(n, _) => *n,
        }
    }

    /// Return the number of active features.
    pub fn n_active(&self) -> usize {
        apply_to_features!(self => activations, {
            activations.iter().filter(|v| v.abs() > 1e-7).count()
        }; indices, {
            indices.len()
        })
    }

    pub fn get(&self, idx: usize) -> Option<&f64> {
        apply_to_features!(self => activations, {
            activations.get(idx)
        }; indices, {
            indices.get(&idx)
        })
    }

    /// Remove one feature entry from the features, if present.
    ///
    /// For the `Features::Dense` variant, the feature is set to zero, and for
    /// the `Features::Sparse` variant, the feature index is removed
    /// entirely.
    ///
    /// ```
    /// use lfa::Features;
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
    /// `Array1`, typically a set of weights.
    ///
    /// ```
    /// use lfa::Features;
    /// use ndarray::Array1;
    ///
    /// let weights = Array1::from_vec(vec![2.0, 5.0, 1.0]);
    ///
    /// assert_eq!(Features::dot(&vec![0.0, 0.2, 0.8].into(), &weights.view()), 1.8);
    /// assert_eq!(Features::dot(&vec![0, 1].into(), &weights.view()), 7.0);
    /// ```
    pub fn dot(&self, weights: &ArrayView1<ActivationT>) -> f64 {
        apply_to_features!(self => activations, {
            Features::dot_dense(activations, weights)
        }; indices, {
            Features::dot_sparse(indices, weights)
        })
    }

    pub fn dot_dense(activations: &DenseT, weights: &ArrayView1<ActivationT>) -> f64 {
        activations.dot(weights)
    }

    pub fn dot_sparse(indices: &SparseT, weights: &ArrayView1<ActivationT>) -> f64 {
        indices.iter().fold(0.0, |acc, (idx, act)| acc + weights[*idx] * act)
    }

    /// Apply the dot product operation between the `Features` and some other
    /// `Vector`, typically a set of weights.
    ///
    /// ```
    /// extern crate ndarray;
    ///
    /// use lfa::Features;
    /// use ndarray::{Array1, Array2};
    ///
    /// let weights = Array2::from_shape_vec((3, 2), vec![2.0, 5.0, 1.0, 3.0, 1.0, 3.0]).unwrap();
    ///
    /// assert!(
    ///     Features::matmul(&vec![0.1, 0.2, 0.7].into(), &weights.view()).all_close(
    ///         &Array1::from_vec(vec![1.1, 3.2]),
    ///         1e-7 // eps
    ///     )
    /// );
    /// assert_eq!(
    ///     Features::matmul(&vec![0, 1].into(), &weights.view()),
    ///     Array1::from_vec(vec![3.0, 8.0])
    /// );
    /// ```
    pub fn matmul(&self, weights: &ArrayView2<f64>) -> Array1<f64> {
        weights
            .gencolumns()
            .into_iter()
            .map(|col| self.dot(&col))
            .collect()
    }

    /// Expand the features and convert it into a raw, dense vector.
    ///
    /// ```
    /// use lfa::Features;
    ///
    /// assert_eq!(
    ///     Features::expanded(vec![0, 2, 1, 4].into()),
    ///     vec![1.0, 1.0, 1.0, 0.0, 1.0].into()
    /// );
    /// ```
    pub fn expanded(self) -> DenseT {
        match self {
            Features::Dense(activations) => activations,
            Features::Sparse(n, indices) => {
                let mut phi = Array1::zeros(n);

                for (idx, act) in indices.into_iter() {
                    phi[idx] = act;
                }

                phi
            },
        }
    }

    /// Stack two feature vectors together, maintaining sparsity where possible.
    ///
    /// ```
    /// use lfa::Features;
    ///
    /// assert_eq!(
    ///     Features::stack(vec![0.0, 1.0].into(), vec![1.0, 0.0, 1.0].into()),
    ///     vec![0.0, 1.0, 1.0, 0.0, 1.0].into()
    /// );
    /// ```
    pub fn stack(self, other: Features) -> Features {
        match (self, other) {
            (Features::Sparse(n1, mut indices_1), Features::Sparse(n2, indices_2)) => {
                indices_2.iter().for_each(|(i, &v)| {
                    indices_1.insert(i + n1, v);
                });

                Features::Sparse(n1 + n2, indices_1)
            },
            (f1, f2) => {
                let mut all_activations = f1.expanded().into_raw_vec();
                all_activations.extend_from_slice(f2.expanded().as_slice().unwrap());

                Features::Dense(all_activations.into())
            },
        }
    }

    pub fn mut_activations(&mut self, f: impl Fn(ActivationT) -> ActivationT) {
        match self {
            Features::Dense(activations) => activations.mapv_inplace(&f),
            Features::Sparse(_, indices) => for a in indices.values_mut() {
                *a = f(*a);
            }
        }
    }

    /// Apply the function `f` to the features if the `Dense` variant or
    /// return `None`.
    pub fn map_dense<T>(self, f: impl FnOnce(DenseT) -> T) -> Option<T> {
        apply_to_features!(self => activations, {
            Some(f(activations))
        }; _indices, {
            None
        })
    }

    /// Apply the function `f` to the features if the `Sparse` variant or
    /// return `None`.
    pub fn map_sparse<T>(self, f: impl FnOnce(SparseT) -> T) -> Option<T> {
        apply_to_features!(self => _activations, {
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

    pub fn combine(self, other: &Features, f: impl Fn(ActivationT, ActivationT) -> ActivationT) -> Features {
        use Features::*;

        match self {
            Dense(mut a1) => match other {
                Dense(a2) => {
                    a1.zip_mut_with(&a2, |x, y| *x = f(*x, *y));
                    Dense(a1)
                },
                Sparse(_, a2) => {
                    for (&i, a) in a2 {
                        a1[i] = f(a1[i], *a);
                    }

                    Dense(a1)
                },
            },
            Sparse(n1, mut a1) => match other {
                Dense(a2) => {
                    let mut phi = unsafe { Array1::uninitialized(n1) };

                    for i in 0..n1 {
                        phi[i] = f(a1.get(&i).cloned().unwrap_or(0.0), a2[i]);
                    }

                    Dense(phi)
                },
                Sparse(n2, a2) => {
                    for (i, a) in a1.iter_mut() {
                        *a = f(*a, a2.get(i).cloned().unwrap_or(0.0));
                    }

                    for (&i, y) in a2.iter() {
                        a1.entry(i).or_insert_with(|| f(0.0, *y));
                    }

                    Sparse(n1.max(*n2), a1)
                },
            }
        }
    }

    pub fn fold(&self, init: f64, f: impl Fn(f64, &f64) -> f64) -> f64 {
        apply_to_features!(self => ref activations, {
            activations.iter().fold(init, f)
        }; ref indices, {
            indices.values().fold(init, f)
        })
    }

    pub fn addto(&self, weights: &mut ArrayViewMut1<ActivationT>) {
        apply_to_features!(self => ref activations, {
            weights.add_assign(activations)
        }; ref indices, {
            for (&idx, act) in indices { weights[idx] += act; }
        });
    }

    pub fn scaled_addto(&self, alpha: ActivationT, weights: &mut ArrayViewMut1<ActivationT>) {
        apply_to_features!(self => ref activations, {
            weights.scaled_add(alpha, activations)
        }; ref indices, {
            for (&idx, act) in indices { weights[idx] += alpha * act; }
        });
    }
}

impl Index<usize> for Features {
    type Output = f64;

    fn index(&self, idx: usize) -> &f64 {
        apply_to_features!(self => activations, {
            activations.index(idx)
        }; indices, {
            if indices.contains_key(&idx) {
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
            (&SparseFeatures(n1, ref idx1), &SparseFeatures(n2, ref idx2)) => {
                n1 == n2 && idx1.eq(idx2)
            },
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
        DenseFeatures(Array1::from_vec(activations))
    }
}

impl FromIterator<ActivationT> for Features {
    fn from_iter<I: IntoIterator<Item = ActivationT>>(iter: I) -> Self {
        DenseFeatures(Array1::from_iter(iter))
    }
}

impl From<SparseT> for Features {
    fn from(indices: SparseT) -> Features {
        let n = indices.iter().map(|(i, _)| i).max().unwrap() + 1;

        SparseFeatures(n, indices)
    }
}

impl From<Vec<IndexT>> for Features {
    fn from(indices: Vec<IndexT>) -> Features {
        let n = indices.iter().max().unwrap() + 1;

        SparseFeatures(n, indices.into_iter().map(|i| (i, 1.0)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sparse(n: usize, indices: impl IntoIterator<Item = IndexT>) -> Features {
        Features::Sparse(n, indices.into_iter().map(|i| (i, 1.0)).collect())
    }

    #[test]
    fn test_sparse() {
        let mut f = make_sparse(11, vec![0usize, 5usize, 10usize]);

        assert!(f.is_sparse());
        assert!(!f.is_dense());
        assert_eq!(f.n_active(), 3);

        f.remove(1);
        assert_eq!(f, make_sparse(11, vec![0usize, 5usize, 10usize]));

        assert_eq!(f.dot(&Array1::ones(11).view()), 3.0);
        assert_eq!(f.dot(&Array1::zeros(11).view()), 0.0);

        f.remove(5);
        assert_eq!(f, make_sparse(11, vec![0usize, 10usize]));

        assert_eq!(f.dot(&Array1::ones(11).view()), 2.0);
        assert_eq!(f.dot(&Array1::zeros(11).view()), 0.0);

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
        assert_eq!(f.n_active(), 4);

        f.remove(10);
        assert_eq!(f, Features::from(vec![0.0, 0.1, 0.2, 0.1, 0.0, 0.6]));

        assert_eq!(f.dot(&Array1::ones(6).view()), 1.0);
        assert_eq!(f.dot(&Array1::zeros(6).view()), 0.0);

        f.remove(1);
        assert_eq!(f, Features::from(vec![0.0, 0.0, 0.2, 0.1, 0.0, 0.6]));

        assert_eq!(f.dot(&Array1::ones(6).view()), 0.9);
        assert_eq!(f.dot(&Array1::zeros(6).view()), 0.0);

        assert_eq!(f[0], 0.0);
        assert_eq!(f[1], 0.0);
        assert_eq!(f[2], 0.2);
        assert_eq!(f[3], 0.1);
        assert_eq!(f[4], 0.0);
        assert_eq!(f[5], 0.6);
    }
}
