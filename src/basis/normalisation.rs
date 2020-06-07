use super::*;
use crate::{ActivationT, Features, IndexT, Result, SparseActivations};
use spaces::{Card, Dim, Space};
use std::f64;

/// Apply _L₀_ normalisation to the output of a `Basis` instance.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct L0Normaliser<B>(B);

impl<B> L0Normaliser<B> {
    pub fn new(basis: B) -> Self { L0Normaliser(basis) }
}

impl<B: Space> Space for L0Normaliser<B> {
    type Value = Features;

    fn dim(&self) -> Dim { self.0.dim() }

    fn card(&self) -> Card { self.0.card() }
}

impl<T, B> Basis<T> for L0Normaliser<B>
where B: Basis<T, Value = Features>
{
    fn project(&self, input: T) -> Result<Features> {
        self.0.project(input).map(|f| {
            let z = f.n_active() as f64;

            f.map_into(|x| x / z)
        })
    }
}

impl<T, B> EnumerableBasis<T> for L0Normaliser<B>
where
    B: Basis<T, Value = Features>,
    B::Value: Index<usize, Output = ActivationT>,
{
    fn ith(&self, input: T, index: IndexT) -> Result<ActivationT> {
        check_index!(index < self.dim().into() => {
            self.0
                .project(input)
                .map(|f| {
                    let z = f.n_active() as f64;

                    unsafe { f.uget(index).unwrap() / z }
                })
        })
    }
}

impl<B> Combinators for L0Normaliser<B> {}

/// Apply _L₁_ normalisation to the output of a `Basis` instance.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct L1Normaliser<B>(B);

impl<B> L1Normaliser<B> {
    pub fn new(basis: B) -> Self { L1Normaliser(basis) }
}

impl<B: Space> Space for L1Normaliser<B> {
    type Value = Features;

    fn dim(&self) -> Dim { self.0.dim() }

    fn card(&self) -> Card { self.0.card() }
}

impl<T, B> Basis<T> for L1Normaliser<B>
where B: Basis<T, Value = Features>
{
    fn project(&self, input: T) -> Result<Features> {
        self.0.project(input).map(|f| match f {
            Features::Dense(da) => {
                let z = da.fold(0.0, |acc, x| acc + x.abs());

                Features::Dense(da.mapv(|x| x / z))
            },
            Features::Sparse(sa) => {
                let z = sa.iter().fold(0.0, |acc, (_, x)| acc + x.abs());

                Features::Sparse(SparseActivations {
                    dim: sa.dim,
                    activations: sa
                        .activations
                        .into_iter()
                        .map(|(i, x)| (i, x / z))
                        .collect(),
                })
            },
        })
    }
}

impl<T, B> EnumerableBasis<T> for L1Normaliser<B>
where
    B: Basis<T, Value = Features>,
    B::Value: Index<usize, Output = ActivationT>,
{
    fn ith(&self, input: T, index: IndexT) -> Result<ActivationT> {
        check_index!(index < self.dim().into() => {
            self.0.project(input).map(|f| match f {
                Features::Dense(da) => {
                    let z = da.fold(0.0, |acc, x| acc + x.abs());

                    da[index] / z
                },
                Features::Sparse(sa) =>
                    sa.activations[&index] / sa.iter().fold(0.0, |acc, (_, x)| acc + x.abs()),
            })
        })
    }
}

impl<B> Combinators for L1Normaliser<B> {}

/// Apply _L₂_ normalisation to the output of a `Basis` instance.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct L2Normaliser<B>(B);

impl<B> L2Normaliser<B> {
    pub fn new(basis: B) -> Self { L2Normaliser(basis) }
}

impl<B: Space> Space for L2Normaliser<B> {
    type Value = Features;

    fn dim(&self) -> Dim { self.0.dim() }

    fn card(&self) -> Card { self.0.card() }
}

impl<T, B> Basis<T> for L2Normaliser<B>
where B: Basis<T, Value = Features>
{
    fn project(&self, input: T) -> Result<Features> {
        self.0.project(input).map(|f| match f {
            Features::Dense(da) => {
                let z = da.fold(0.0, |acc, x| acc + x * x).sqrt();

                Features::Dense(da.mapv(|x| x / z))
            },
            Features::Sparse(sa) => {
                let z = sa.iter().fold(0.0, |acc, (_, x)| acc + x * x).sqrt();

                Features::Sparse(SparseActivations {
                    dim: sa.dim,
                    activations: sa
                        .activations
                        .into_iter()
                        .map(|(i, x)| (i, x / z))
                        .collect(),
                })
            },
        })
    }
}

impl<T, B> EnumerableBasis<T> for L2Normaliser<B>
where
    B: Basis<T, Value = Features>,
    B::Value: Index<usize, Output = ActivationT>,
{
    fn ith(&self, input: T, index: IndexT) -> Result<ActivationT> {
        check_index!(index < self.dim().into() => {
            self.0.project(input).map(|f| match f {
                Features::Dense(da) => {
                    let z = da.fold(0.0, |acc, x| acc + x * x).sqrt();

                    da[index] / z
                },
                Features::Sparse(sa) =>
                    sa.activations[&index] / sa.iter().fold(0.0, |acc, (_, x)| acc + x * x).sqrt(),
            })
        })
    }
}

impl<B> Combinators for L2Normaliser<B> {}

/// Apply _L∞_ normalisation to the output of a `Basis` instance.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct LinfNormaliser<B>(B);

impl<B> LinfNormaliser<B> {
    pub fn new(basis: B) -> Self { LinfNormaliser(basis) }
}

impl<B: Space> Space for LinfNormaliser<B> {
    type Value = Features;

    fn dim(&self) -> Dim { self.0.dim() }

    fn card(&self) -> Card { self.0.card() }
}

impl<T, B> Basis<T> for LinfNormaliser<B>
where B: Basis<T, Value = Features>
{
    fn project(&self, input: T) -> Result<Features> {
        self.0.project(input).map(|f| match f {
            Features::Dense(da) => {
                let z = da.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x.abs()));

                Features::Dense(da.mapv(|x| x / z))
            },
            Features::Sparse(sa) => {
                let z = sa
                    .iter()
                    .fold(f64::NEG_INFINITY, |acc, (_, x)| acc.max(x.abs()));

                Features::Sparse(SparseActivations {
                    dim: sa.dim,
                    activations: sa
                        .activations
                        .into_iter()
                        .map(|(i, x)| (i, x / z))
                        .collect(),
                })
            },
        })
    }
}

impl<T, B> EnumerableBasis<T> for LinfNormaliser<B>
where
    B: Basis<T, Value = Features>,
    B::Value: Index<usize, Output = ActivationT>,
{
    fn ith(&self, input: T, index: IndexT) -> Result<ActivationT> {
        check_index!(index < self.dim().into() => {
            self.0.project(input).map(|f| match f {
                Features::Dense(da) => {
                    let z = da.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x.abs()));

                    da[index] / z
                },
                Features::Sparse(sa) => sa.activations[&index] / sa.iter().fold(
                    f64::NEG_INFINITY,
                    |acc, (_, x)| acc.max(x.abs())
                ),
            })
        })
    }
}

impl<B> Combinators for LinfNormaliser<B> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::Fixed;

    quickcheck! {
        fn test_l1(constants: Vec<f64>) -> bool {
            let p = L1Normaliser::new(Fixed::dense(constants.clone()));
            let f = p.project(&[0.0]).unwrap().into_dense();

            let abssum: f64 = constants.iter().map(|v| v.abs()).sum();

            f.into_iter().zip(constants.into_iter()).all(|(x, y)| (x - y / abssum) < 1e-7)
        }
    }

    quickcheck! {
        fn test_l2(constants: Vec<f64>) -> bool {
            let p = L2Normaliser::new(Fixed::dense(constants.clone()));
            let f = p.project(&[0.0]).unwrap().into_dense();

            let sqsum: f64 = constants.iter().map(|v| v*v).sum();

            f.into_iter().zip(constants.into_iter()).all(|(x, y)| (x - y / sqsum.sqrt()) < 1e-7)
        }
    }
}
