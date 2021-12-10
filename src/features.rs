#![macro_use]
use ndarray::{linalg::Dot, Array1, ArrayBase, Data, DataMut, Dimension, NdIndex};
use std::{
    iter::FromIterator,
    mem::{self, MaybeUninit},
    ops::{AddAssign, Index},
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
    ($features:expr => ref $dense:ident, $dbody:block; ref $sparse:ident, $sbody:block) => {
        match $features {
            Features::Dense(ref $dense) => $dbody,
            Features::Sparse(ref $sparse) => $sbody,
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

/// Internal type used to represent a feature's index.
pub type IndexT = usize;

/// Internal type used to represent a feature's activation.
pub type ActivationT = f64;

/// Internal type used to represent a dense feature vector.
pub type DenseActivations = Array1<ActivationT>;

/// Internal type used to represent a sparse feature vector.
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct SparseActivations {
    /// The total number of features.
    pub dim: usize,

    /// The subset of active features and their activations.
    pub activations: ::std::collections::HashMap<IndexT, ActivationT>,
}

impl SparseActivations {
    /// An iterator visiting all feature-activation pairs in arbitrary order.
    /// The iterator element type is `(&'a IndexT, &'a ActivationT)`.
    pub fn iter(&self) -> ::std::collections::hash_map::Iter<IndexT, ActivationT> {
        self.activations.iter()
    }
}

// TODO: This is all quite inefficient. We should use combinators for features.
// Once we have an autograd crate we should be able to implement this trivially
// byt having each Basis define it's own custom `Buffer` type.
/// Sparse/dense feature vector representation.
///
/// __Note:__ many of the methods associated with `Features` are based on those
/// of the `ArrayBase` type provided in the [`ndarray`] crate.
///
/// [`ndarray`]: https://crates.io/crates/ndarray
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub enum Features {
    /// Dense, floating-point activation vector.
    Dense(DenseActivations),

    /// Sparse, index-based activation vector.
    Sparse(SparseActivations),
}

pub(crate) use self::Features::{Dense as DenseFeatures, Sparse as SparseFeatures};

impl Features {
    /// Construct a `Dense` feature vector from an iterable collection of
    /// activations.
    pub fn dense<I>(da: I) -> Features
    where
        I: IntoIterator<Item = ActivationT>,
    {
        Features::Dense(da.into_iter().collect())
    }

    /// Construct a `Sparse` feature vector with given dimensionality from an
    /// iterable collection of feature-activation pairs.
    pub fn sparse<I>(dim: usize, activations: I) -> Features
    where
        I: IntoIterator<Item = (IndexT, ActivationT)>,
    {
        Features::Sparse(SparseActivations {
            dim,
            activations: activations.into_iter().collect(),
        })
    }

    /// Construct a `Sparse` feature vector with given dimensionality from an
    /// iterable collection of feature indices.
    pub fn unitary<I>(dim: usize, indices: I) -> Features
    where
        I: IntoIterator<Item = IndexT>,
    {
        Features::sparse(dim, indices.into_iter().map(|i| (i, 1.0)))
    }

    /// Return true if the features is the `Dense` variant.
    pub fn is_dense(&self) -> bool {
        apply_to_features!(self => true; false)
    }

    /// Return true if the features is the `Sparse` variant.
    pub fn is_sparse(&self) -> bool {
        apply_to_features!(self => false; true)
    }

    /// Return the number of features.
    pub fn n_features(&self) -> usize {
        match self {
            Features::Dense(ref da) => da.len(),
            Features::Sparse(sa) => sa.dim,
        }
    }

    /// Return the number of _active_ features.
    pub fn n_active(&self) -> usize {
        apply_to_features!(self => da, {
            da.iter().filter(|v| v.abs() > 1e-7).count()
        }; sa, {
            sa.activations.len()
        })
    }

    /// Return the activation of the feature at index `idx`, if defined.
    ///
    /// __Panics__ if the index exceeds the size of the feature vector.
    pub fn get(&self, idx: usize) -> Option<&f64> {
        apply_to_features!(self => da, {
            da.get(idx)
        }; sa, {
            sa.activations.get(&idx)
        })
    }

    /// Return the activation of the feature at index `idx` _without bounds
    /// checking_.
    pub unsafe fn uget(&self, idx: usize) -> Option<&f64> {
        apply_to_features!(self => da, {
            Some(da.uget(idx))
        }; sa, {
            sa.activations.get(&idx)
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
    /// let mut dense = Features::dense(vec![0.0, 0.2, 0.4, 0.4]);
    /// let mut sparse = Features::unitary(16, vec![0, 10, 15]);
    ///
    /// dense.remove(1);
    /// sparse.remove(10);
    ///
    /// assert_eq!(dense, vec![0.0, 0.0, 0.4, 0.4].into());
    /// assert_eq!(sparse, vec![0, 15].into());
    /// ```
    pub fn remove(&mut self, idx: usize) {
        apply_to_features!(self => mut da, {
            if let Some(a) = da.get_mut(idx) {
                *a = 0.0;
            }
        }; mut sa, {
            sa.activations.remove(&idx);
        })
    }

    /// Clone the features and convert into a raw, dense vector of activations.
    pub fn to_dense(&self) -> DenseActivations {
        match self {
            Features::Dense(da) => da.clone(),
            Features::Sparse(sa) => {
                let mut phi = Array1::zeros(sa.dim);

                for (&idx, &act) in sa.iter() {
                    phi[idx] = act;
                }

                phi
            }
        }
    }

    /// Expand the features directly into a raw, dense vector of activations.
    ///
    /// ```
    /// use lfa::Features;
    /// use ndarray::Array1;
    ///
    /// let phi = Features::unitary(5, vec![0, 2, 1, 4]);
    ///
    /// assert_eq!(phi.into_dense(), Array1::from(vec![1.0, 1.0, 1.0, 0.0, 1.0]));
    /// ```
    pub fn into_dense(self) -> DenseActivations {
        match self {
            Features::Dense(da) => da,
            Features::Sparse(sa) => {
                let mut phi = Array1::zeros(sa.dim);

                for (idx, act) in sa.activations.into_iter() {
                    phi[idx] = act;
                }

                phi
            }
        }
    }

    /// Stack two feature vectors together, maintaining sparsity if possible.
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
        use Features::*;

        match (self, other) {
            (Sparse(mut sa1), Sparse(sa2)) => {
                sa2.activations.iter().for_each(|(i, &v)| {
                    sa1.activations.insert(i + sa1.dim, v);
                });

                Sparse(SparseActivations {
                    dim: sa1.dim + sa2.dim,
                    activations: sa1.activations,
                })
            }
            (f1, f2) => {
                let mut all_activations = f1.into_dense().into_raw_vec();
                all_activations.extend_from_slice(f2.into_dense().as_slice().unwrap());

                Dense(all_activations.into())
            }
        }
    }

    /// Map all feature activations using an operation, `f`, and return a new
    /// `Features` instance.
    pub fn map(&self, f: impl Fn(ActivationT) -> ActivationT) -> Features {
        use Features::*;

        match self {
            Dense(da) => Dense(da.mapv(&f)),

            // Is the function sparsity preserving?
            Sparse(sa) => {
                let null_activation = f(0.0);

                if null_activation.abs() < 1e-5 {
                    let activations = sa.iter().map(|(&i, &k)| (i, f(k))).collect();

                    Sparse(SparseActivations {
                        dim: sa.dim,
                        activations,
                    })
                } else {
                    let mut phi = Array1::from_elem(sa.dim, null_activation);

                    for (&idx, &act) in sa.iter() {
                        phi[idx] = f(act);
                    }

                    Dense(phi)
                }
            }
        }
    }

    /// Map all feature activations using an operation, `f`, and return.
    pub fn map_into(self, f: impl Fn(ActivationT) -> ActivationT) -> Features {
        use Features::*;

        match self {
            Dense(activations) => Dense(activations.mapv_into(&f)),

            // Is the function sparsity preserving?
            Sparse(sa) => {
                let null_activation = f(0.0);

                if null_activation.abs() < 1e-5 {
                    let activations = sa.activations.into_iter().map(|(i, v)| (i, f(v))).collect();

                    Sparse(SparseActivations {
                        dim: sa.dim,
                        activations,
                    })
                } else {
                    let mut phi = Array1::from_elem(sa.dim, null_activation);

                    for (idx, act) in sa.activations.into_iter() {
                        phi[idx] = f(act);
                    }

                    Dense(phi)
                }
            }
        }
    }

    /// Mutate all feature activations inplace using an operation, `f`.
    pub fn map_inplace(&mut self, f: impl Fn(ActivationT) -> ActivationT) {
        let tmp = unsafe { mem::MaybeUninit::zeroed().assume_init() };
        let old = mem::replace(self, tmp);

        mem::drop(mem::replace(self, old.map_into(f)));
    }

    /// Map the function `f` over the internal `DenseActivations` representation
    /// if `self` is the `Dense` variant, otherwise return `None`.
    pub fn map_dense<T>(self, f: impl FnOnce(DenseActivations) -> T) -> Option<T> {
        apply_to_features!(self => activations, {
            Some(f(activations))
        }; _indices, {
            None
        })
    }

    /// Map the function `f` over the internal `SparseActivations`
    /// representation if `self` is the `Dense` variant, otherwise return
    /// `None`.
    pub fn map_sparse<T>(self, f: impl FnOnce(SparseActivations) -> T) -> Option<T> {
        apply_to_features!(self => _activations, {
            None
        }; indices, {
            Some(f(indices))
        })
    }

    /// Map the function `f_dense` or `f_sparse` based on the sparsity of the
    /// features.
    pub fn map_either<T>(
        self,
        f_dense: impl FnOnce(DenseActivations) -> T,
        f_sparse: impl FnOnce(SparseActivations) -> T,
    ) -> T {
        apply_to_features!(self => activations, {
            f_dense(activations)
        }; indices, {
            f_sparse(indices)
        })
    }

    /// Merge `self` with another feature vector and an operation, `f`,
    /// returning a new instance.
    pub fn merge(
        &self,
        other: &Features,
        f: impl Fn(ActivationT, ActivationT) -> ActivationT,
    ) -> Features {
        use Features::*;

        match self {
            Dense(da1) => {
                let mut a_out = da1.clone();

                match other {
                    Dense(da2) => {
                        a_out.zip_mut_with(da2, |x, y| *x = f(*x, *y));

                        Dense(a_out)
                    }
                    Sparse(sa2) => {
                        for (&i, a) in sa2.iter() {
                            a_out[i] = f(a_out[i], *a);
                        }

                        Dense(a_out)
                    }
                }
            }
            Sparse(sa1) => match other {
                Dense(da2) => {
                    let n = sa1.dim.max(da2.len());
                    let mut phi = Array1::uninit(n);

                    for i in 0..n {
                        phi[i] = MaybeUninit::new(f(
                            sa1.activations.get(&i).cloned().unwrap_or(0.0),
                            da2[i],
                        ));
                    }

                    unsafe { Dense(phi.assume_init()) }
                }
                Sparse(sa2) => {
                    let mut idx_out = sa1.activations.clone();

                    for (i, a) in idx_out.iter_mut() {
                        *a = f(*a, sa2.activations.get(i).cloned().unwrap_or(0.0));
                    }

                    for (&i, y) in sa2.activations.iter() {
                        idx_out.entry(i).or_insert_with(|| f(0.0, *y));
                    }

                    Sparse(SparseActivations {
                        dim: sa1.dim.max(sa2.dim),
                        activations: idx_out,
                    })
                }
            },
        }
    }

    /// Merge `self` with another feature vector using a given operation, `f`.
    pub fn merge_into(
        self,
        other: &Features,
        f: impl Fn(ActivationT, ActivationT) -> ActivationT,
    ) -> Features {
        use Features::*;

        match self {
            Dense(mut da1) => match other {
                Dense(sa2) => {
                    da1.zip_mut_with(sa2, |x, y| *x = f(*x, *y));

                    Dense(da1)
                }
                Sparse(sa2) => {
                    for (&i, a) in sa2.iter() {
                        da1[i] = f(da1[i], *a);
                    }

                    Dense(da1)
                }
            },
            Sparse(mut sa1) => match other {
                Dense(da2) => {
                    let n = sa1.dim.max(da2.len());
                    let mut phi = Array1::uninit(n);

                    for i in 0..n {
                        phi[i] = MaybeUninit::new(f(
                            sa1.activations.get(&i).copied().unwrap_or(0.0),
                            da2.get(i).copied().unwrap_or(0.0),
                        ));
                    }

                    unsafe { Dense(phi.assume_init()) }
                }
                Sparse(sa2) => {
                    for (i, a) in sa1.activations.iter_mut() {
                        *a = f(*a, sa2.activations.get(i).cloned().unwrap_or(0.0));
                    }

                    for (&i, y) in sa2.activations.iter() {
                        sa1.activations.entry(i).or_insert_with(|| f(0.0, *y));
                    }

                    Sparse(SparseActivations {
                        dim: sa1.dim.max(sa2.dim),
                        activations: sa1.activations,
                    })
                }
            },
        }
    }

    /// Merge `self` in-place with another feature vector using an operation,
    /// `f`.
    pub fn merge_inplace(
        &mut self,
        other: &Features,
        f: impl Fn(ActivationT, ActivationT) -> ActivationT,
    ) {
        let tmp = unsafe { mem::MaybeUninit::zeroed().assume_init() };
        let old = mem::replace(self, tmp);

        mem::drop(mem::replace(self, old.merge_into(other, f)));
    }

    /// Perform a fold operation over the feature activations.
    ///
    /// __Note:__ for sparse features this method will ignore zeroes.
    pub fn fold(&self, init: f64, f: impl Fn(f64, &f64) -> f64) -> f64 {
        apply_to_features!(self => ref da, {
            da.iter().fold(init, f)
        }; ref sa, {
            sa.activations.values().fold(init, f)
        })
    }

    /// Apply the dot product operation between the `Features` and some other
    /// `Array1`, typically a set of weights.
    ///
    /// ```
    /// use lfa::Features;
    /// use ndarray::Array1;
    ///
    /// let weights = Array1::from(vec![2.0, 5.0, 1.0]);
    ///
    /// assert_eq!(Features::dot(&vec![0.0, 0.2, 0.8].into(), &weights.view()), 1.8);
    /// assert_eq!(Features::dot(&vec![0, 1].into(), &weights.view()), 7.0);
    /// ```
    pub fn dot<W>(&self, weights: &W) -> f64
    where
        W: std::ops::Index<usize, Output = f64>,
        DenseActivations: Dot<W, Output = f64>,
    {
        apply_to_features!(self => activations, {
            Features::dot_dense(activations, weights)
        }; indices, {
            Features::dot_sparse(indices, weights)
        })
    }

    fn dot_dense<W>(da: &DenseActivations, weights: &W) -> f64
    where
        DenseActivations: Dot<W, Output = f64>,
    {
        da.dot(weights)
    }

    fn dot_sparse<W>(sa: &SparseActivations, weights: &W) -> f64
    where
        W: std::ops::Index<usize, Output = f64>,
    {
        sa.activations
            .iter()
            .fold(0.0, |acc, (idx, act)| acc + weights[*idx] * act)
    }

    /// Apply the matrix multiplication operation between the `Features` and a
    /// `Matrix`.
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
    ///         &Array1::from(vec![1.1, 3.2]),
    ///         1e-7 // eps
    ///     )
    /// );
    /// assert_eq!(
    ///     Features::matmul(&vec![0, 1].into(), &weights.view()),
    ///     Array1::from(vec![3.0, 8.0])
    /// );
    /// ```
    pub fn matmul<S>(&self, weights: &ArrayBase<S, ndarray::Ix2>) -> Array1<f64>
    where
        S: Data<Elem = f64>,
    {
        weights
            .columns()
            .into_iter()
            .map(|col| self.dot(&col))
            .collect()
    }

    /// Returns the sum of all activations in the feature vector.
    pub fn sum(&self) -> ActivationT {
        match self {
            Features::Dense(da) => da.sum(),
            Features::Sparse(sa) => sa.activations.values().sum(),
        }
    }

    /// Perform an elementwise add of activations to a `weights` vector.
    pub fn addto<S, E>(&self, weights: &mut ArrayBase<S, E>)
    where
        S: DataMut<Elem = f64>,
        E: Dimension,
        usize: NdIndex<E>,
    {
        apply_to_features!(self => ref da, {
            weights.add_assign(da)
        }; ref sa, {
            for (&idx, act) in sa.iter() { weights[idx] += act; }
        });
    }

    /// Perform an elementwise add of activations (scaled by `alpha`) to a
    /// `weights` vector.
    pub fn scaled_addto<S, E>(&self, alpha: ActivationT, weights: &mut ArrayBase<S, E>)
    where
        S: DataMut<Elem = f64>,
        E: Dimension,
        usize: NdIndex<E>,
    {
        apply_to_features!(self => ref da, {
            weights.scaled_add(alpha, da)
        }; ref sa, {
            for (&idx, act) in sa.iter() { weights[idx] += alpha * act; }
        });
    }
}

impl<W> ndarray::linalg::Dot<W> for Features
where
    W: std::ops::Index<usize, Output = f64>,
    DenseActivations: Dot<W, Output = f64>,
{
    type Output = f64;

    fn dot(&self, rhs: &W) -> f64 {
        self.dot(rhs)
    }
}

impl ndarray::linalg::Dot<Features> for Features {
    type Output = f64;

    fn dot(&self, rhs: &Features) -> f64 {
        use Features::*;

        match self {
            Sparse(sa1) => match rhs {
                Sparse(sa2) => sa1.activations.keys().fold(0, |acc, k| {
                    acc + if sa2.activations.contains_key(k) {
                        1
                    } else {
                        0
                    }
                }) as f64,
                Dense(da2) => Features::dot_sparse(sa1, da2),
            },
            Dense(da1) => match rhs {
                Sparse(sa2) => Features::dot_sparse(sa2, da1),
                Dense(da2) => da1.dot(da2),
            },
        }
    }
}

impl AsRef<Features> for Features {
    fn as_ref(&self) -> &Features {
        self
    }
}

impl Index<usize> for Features {
    type Output = f64;

    fn index(&self, idx: usize) -> &f64 {
        apply_to_features!(self => da, {
            da.index(idx)
        }; sa, {
            if sa.activations.contains_key(&idx) {
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
            (SparseFeatures(sa1), SparseFeatures(sa2)) => {
                sa1.dim == sa2.dim && sa1.activations.eq(&sa2.activations)
            }
            (&DenseFeatures(ref da1), &DenseFeatures(ref da2)) => da1.eq(&da2),
            _ => unimplemented!(
                "Cannot check equality of dense/sparse with no knowledge of the \
                 full dimensionality of sparse features."
            ),
        }
    }
}

impl From<DenseActivations> for Features {
    fn from(activations: DenseActivations) -> Features {
        DenseFeatures(activations)
    }
}

impl From<Vec<ActivationT>> for Features {
    fn from(activations: Vec<ActivationT>) -> Features {
        DenseFeatures(Array1::from(activations))
    }
}

impl FromIterator<ActivationT> for Features {
    fn from_iter<I: IntoIterator<Item = ActivationT>>(iter: I) -> Self {
        DenseFeatures(Array1::from_iter(iter))
    }
}

impl From<SparseActivations> for Features {
    fn from(sa: SparseActivations) -> Features {
        let n = sa.activations.iter().map(|(i, _)| i).max().unwrap() + 1;

        SparseFeatures(SparseActivations {
            dim: n,
            activations: sa.activations,
        })
    }
}

impl From<Vec<IndexT>> for Features {
    fn from(indices: Vec<IndexT>) -> Features {
        let n = indices.iter().max().unwrap() + 1;

        SparseFeatures(SparseActivations {
            dim: n,
            activations: indices.into_iter().map(|i| (i, 1.0)).collect(),
        })
    }
}

impl From<Features> for DenseActivations {
    fn from(phi: Features) -> DenseActivations {
        phi.into_dense()
    }
}

impl From<Features> for Vec<ActivationT> {
    fn from(phi: Features) -> Vec<ActivationT> {
        phi.into_dense().into_raw_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sparse(n: usize, indices: impl IntoIterator<Item = IndexT>) -> Features {
        Features::Sparse(SparseActivations {
            dim: n,
            activations: indices.into_iter().map(|i| (i, 1.0)).collect(),
        })
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
        let mut f = Features::dense(vec![0.0, 0.1, 0.2, 0.1, 0.0, 0.6]);

        assert!(f.is_dense());
        assert!(!f.is_sparse());
        assert_eq!(f.n_active(), 4);

        f.remove(10);
        assert_eq!(f, Features::dense(vec![0.0, 0.1, 0.2, 0.1, 0.0, 0.6]));

        assert_eq!(f.dot(&Array1::ones(6).view()), 1.0);
        assert_eq!(f.dot(&Array1::zeros(6).view()), 0.0);

        f.remove(1);
        assert_eq!(f, Features::dense(vec![0.0, 0.0, 0.2, 0.1, 0.0, 0.6]));

        assert_eq!(f.dot(&Array1::ones(6).view()), 0.9);
        assert_eq!(f.dot(&Array1::zeros(6).view()), 0.0);

        assert_eq!(f[0], 0.0);
        assert_eq!(f[1], 0.0);
        assert_eq!(f[2], 0.2);
        assert_eq!(f[3], 0.1);
        assert_eq!(f[4], 0.0);
        assert_eq!(f[5], 0.6);
    }

    #[test]
    fn test_map_inplace() {
        let mut f = Features::unitary(11, vec![0usize, 5usize, 10usize]);

        f.map_inplace(|a| a);
        assert!(f.is_sparse());
        assert_eq!(f.sum(), 3.0);

        f.map_inplace(|a| a * 2.0);
        assert!(f.is_sparse());
        assert_eq!(f.sum(), 6.0);

        f.map_inplace(|_| 1.0);
        assert!(f.is_dense());
        assert_eq!(f.sum(), 11.0);

        f.map_inplace(|a| a);
        assert!(f.is_dense());
        assert_eq!(f.sum(), 11.0);
    }

    #[test]
    fn test_merge_inplace() {
        let mut f1 = Features::unitary(11, vec![0usize, 5usize, 10usize]);
        let f2 = Features::dense(vec![0.0, 0.1, 0.2, 0.1, 0.0, 0.6]);

        f1.merge_inplace(&f2, |x, y| x + y);
        assert!(f1.is_dense());

        assert_eq!(f1[0], 1.0);
        assert_eq!(f1[1], 0.1);
        assert_eq!(f1[2], 0.2);
        assert_eq!(f1[3], 0.1);
        assert_eq!(f1[4], 0.0);
        assert_eq!(f1[5], 1.6);
        assert_eq!(f1[6], 0.0);
        assert_eq!(f1[7], 0.0);
        assert_eq!(f1[8], 0.0);
        assert_eq!(f1[9], 0.0);
        assert_eq!(f1[10], 1.0);
    }
}
