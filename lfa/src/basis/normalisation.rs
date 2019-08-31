use crate::{IndexT, ActivationT, Result, check_index, Features, basis::Projector};
use std::f64;

/// Apply _L₀_ normalisation to the output of a `Projector` instance.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct L0Normaliser<P>(P);

impl<P> L0Normaliser<P> {
    pub fn new(projector: P) -> Self {
        L0Normaliser(projector)
    }
}

impl<P: Projector> Projector for L0Normaliser<P> {
    fn n_features(&self) -> usize {
        self.0.n_features()
    }

    fn project_ith(&self, input: &[f64], index: IndexT) -> Result<Option<ActivationT>> {
        check_index(index, self.n_features(), || {
            self.0.project(input)
                .map(|f| {
                    let z = f.n_active() as f64;

                    unsafe { f.uget(index).cloned().map(|x| x / z) }
                })
        })
    }

    fn project(&self, input: &[f64]) -> Result<Features> {
        self.0.project(input)
            .map(|mut f| {
                let z = f.n_active() as f64;
                f.mut_activations(|x| x / z);
                f
            })
    }
}

/// Apply _L₁_ normalisation to the output of a `Projector` instance.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct L1Normaliser<P>(P);

impl<P> L1Normaliser<P> {
    pub fn new(projector: P) -> Self {
        L1Normaliser(projector)
    }
}

impl<P: Projector> Projector for L1Normaliser<P> {
    fn n_features(&self) -> usize {
        self.0.n_features()
    }

    fn project_ith(&self, input: &[f64], index: IndexT) -> Result<Option<ActivationT>> {
        check_index(index, self.n_features(), || {
            self.0.project(input).map(|f| match f {
                Features::Dense(activations) => {
                    let z = activations.fold(0.0, |acc, x| acc + x.abs());

                    Some(activations[index] / z)
                },
                Features::Sparse(_, indices) => indices.get(&index).cloned().map(|f| {
                    f / indices.iter().fold(0.0, |acc, (_, x)| acc + x.abs())
                }),
            })
        })
    }

    fn project(&self, input: &[f64]) -> Result<Features> {
        self.0.project(input).map(|f| match f {
            Features::Dense(activations) => {
                let z = activations.fold(0.0, |acc, x| acc + x.abs());

                Features::Dense(activations.mapv(|x| x / z))
            },
            Features::Sparse(n, indices) => {
                let z = indices.iter().fold(0.0, |acc, (_, x)| acc + x.abs());

                Features::Sparse(n, indices.into_iter().map(|(i, x)| (i, x / z)).collect())
            },
        })
    }
}

/// Apply _L₂_ normalisation to the output of a `Projector` instance.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct L2Normaliser<P>(P);

impl<P> L2Normaliser<P> {
    pub fn new(projector: P) -> Self {
        L2Normaliser(projector)
    }
}

impl<P: Projector> Projector for L2Normaliser<P> {
    fn n_features(&self) -> usize { self.0.n_features() }

    fn project_ith(&self, input: &[f64], index: IndexT) -> Result<Option<ActivationT>> {
        check_index(index, self.n_features(), || {
            self.0.project(input).map(|f| match f {
                Features::Dense(activations) => {
                    let z = activations.fold(0.0, |acc, x| acc + x * x).sqrt();

                    Some(activations[index] / z)
                },
                Features::Sparse(_, indices) => indices.get(&index).cloned().map(|f| {
                    f / indices.iter().fold(0.0, |acc, (_, x)| acc + x * x).sqrt()
                }),
            })
        })
    }

    fn project(&self, input: &[f64]) -> Result<Features> {
        self.0.project(input).map(|f| match f {
            Features::Dense(activations) => {
                let z = activations.fold(0.0, |acc, x| acc + x * x).sqrt();

                Features::Dense(activations.mapv(|x| x / z))
            },
            Features::Sparse(n, indices) => {
                let z = indices.iter().fold(0.0, |acc, (_, x)| acc + x * x).sqrt();

                Features::Sparse(n, indices.into_iter().map(|(i, x)| (i, x / z)).collect())
            },
        })
    }
}

/// Apply _L∞_ normalisation to the output of a `Projector` instance.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct LinfNormaliser<P>(P);

impl<P> LinfNormaliser<P> {
    pub fn new(projector: P) -> Self {
        LinfNormaliser(projector)
    }
}

impl<P: Projector> Projector for LinfNormaliser<P> {
    fn n_features(&self) -> usize {
        self.0.n_features()
    }

    fn project_ith(&self, input: &[f64], index: IndexT) -> Result<Option<ActivationT>> {
        check_index(index, self.n_features(), || {
            self.0.project(input).map(|f| match f {
                Features::Dense(activations) => {
                    let z = activations.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x.abs()));

                    Some(activations[index] / z)
                },
                Features::Sparse(_, indices) => indices.get(&index).cloned().map(|f| {
                    f / indices.iter().fold(f64::NEG_INFINITY, |acc, (_, x)| acc.max(x.abs()))
                }),
            })
        })
    }

    fn project(&self, input: &[f64]) -> Result<Features> {
        self.0.project(input).map(|f| match f {
            Features::Dense(activations) => {
                let z = activations.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x.abs()));

                Features::Dense(activations.mapv(|x| x / z))
            },
            Features::Sparse(n, indices) => {
                let z = indices.iter().fold(f64::NEG_INFINITY, |acc, (_, x)| acc.max(x.abs()));

                Features::Sparse(n, indices.into_iter().map(|(i, x)| (i, x / z)).collect())
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::basis::Constants;
    use super::*;

    quickcheck! {
        fn test_l1(constants: Vec<f64>) -> bool {
            let p = L1Normaliser::new(Constants::new(constants.clone()));
            let f = p.project(&[0.0]).unwrap().expanded();

            let abssum: f64 = constants.iter().map(|v| v.abs()).sum();

            f.into_iter().zip(constants.into_iter()).all(|(x, y)| (x - y / abssum) < 1e-7)
        }
    }

    quickcheck! {
        fn test_l2(constants: Vec<f64>) -> bool {
            let p = L2Normaliser::new(Constants::new(constants.clone()));
            let f = p.project(&[0.0]).unwrap().expanded();

            let sqsum: f64 = constants.iter().map(|v| v*v).sum();

            f.into_iter().zip(constants.into_iter()).all(|(x, y)| (x - y / sqsum.sqrt()) < 1e-7)
        }
    }
}
