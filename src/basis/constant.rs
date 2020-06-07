use super::*;
use crate::{ActivationT, Features, IndexT, Result, SparseActivations};
use spaces::{Card, Dim, Space};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Bias(pub f64);

impl Bias {
    pub fn unit() -> Self { Bias(1.0) }
}

impl Space for Bias {
    type Value = Features;

    fn dim(&self) -> Dim { Dim::Finite(1) }

    fn card(&self) -> Card { Card::Infinite }
}

impl<T> Basis<T> for Bias {
    fn project(&self, _: T) -> Result<Features> { Ok(vec![self.0].into()) }
}

impl<T> EnumerableBasis<T> for Bias {
    fn ith(&self, _: T, index: IndexT) -> Result<ActivationT> {
        check_index!(index < 1 => { Ok(self.0) })
    }
}

impl Combinators for Bias {}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Fixed<T>(pub T);

impl<T> Combinators for Fixed<T> {}

impl Fixed<Vec<f64>> {
    pub fn dense<I: IntoIterator<Item = f64>>(constants: I) -> Self {
        Fixed(constants.into_iter().collect())
    }
}

impl Space for Fixed<Vec<f64>> {
    type Value = Features;

    fn dim(&self) -> Dim { Dim::Finite(self.0.len()) }

    fn card(&self) -> Card { Card::Infinite }
}

impl<T> Basis<T> for Fixed<Vec<f64>> {
    fn project(&self, _: T) -> Result<Features> { Ok(self.0.clone().into()) }
}

impl<T> EnumerableBasis<T> for Fixed<Vec<f64>> {
    fn ith(&self, _: T, index: IndexT) -> Result<ActivationT> {
        check_index!(index < 1 => {
            Ok(*self.0.get(index).unwrap())
        })
    }
}

impl Fixed<(usize, HashSet<IndexT>)> {
    pub fn sparse<I>(n_features: usize, active_features: I) -> Self
    where I: IntoIterator<Item = usize> {
        Fixed((n_features, active_features.into_iter().collect()))
    }
}

impl Space for Fixed<(usize, HashSet<IndexT>)> {
    type Value = Features;

    fn dim(&self) -> Dim { Dim::Finite((self.0).0) }

    fn card(&self) -> Card { Card::Infinite }
}

impl<T> Basis<T> for Fixed<(usize, HashSet<IndexT>)> {
    fn project(&self, _: T) -> Result<Features> {
        Ok(Features::Sparse(SparseActivations {
            dim: (self.0).0,
            activations: (self.0).1.iter().map(|&i| (i, 1.0)).collect(),
        }))
    }
}

impl<T> EnumerableBasis<T> for Fixed<(usize, HashSet<IndexT>)> {
    fn ith(&self, _: T, index: IndexT) -> Result<ActivationT> {
        check_index!(index < (self.0).0 => {
            Ok(if (self.0).1.contains(&index) { 1.0 } else { 0.0 })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    quickcheck! {
        fn test_project_general(constants: Vec<f64>, input: Vec<f64>) -> bool {
            let projector = Fixed::dense(constants.clone());
            let activations = projector.project(&input).unwrap().into_dense().into_raw_vec();

            activations.len() == constants.len() && activations == constants
        }
    }
}
