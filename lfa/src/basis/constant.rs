use crate::{Features, IndexT, ActivationT, Result, check_index, basis::Projector};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Constant(f64);

impl Constant {
    pub fn new(value: f64) -> Self { Constant(value) }

    pub fn unit() -> Self { Constant::new(1.0) }
}

impl Projector for Constant {
    fn n_features(&self) -> usize { 1 }

    fn project_ith(&self, _: &[f64], index: IndexT) -> Result<Option<ActivationT>> {
        check_index(index, 1, || Ok(Some(self.0)))
    }

    fn project(&self, _: &[f64]) -> Result<Features> { Ok(vec![self.0].into()) }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Constants(Vec<f64>);

impl Constants {
    pub fn new(constants: Vec<f64>) -> Self { Constants(constants) }
}

impl Projector for Constants {
    fn n_features(&self) -> usize { self.0.len() }

    fn project_ith(&self, _: &[f64], index: IndexT) -> Result<Option<ActivationT>> {
        check_index(index, 1, || Ok(self.0.get(index).cloned()))
    }

    fn project(&self, _: &[f64]) -> Result<Features> { Ok(self.0.clone().into()) }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Indices {
    n_features: usize,
    active_features: HashSet<IndexT>,
}

impl Indices {
    pub fn new(n_features: usize, active_features: Vec<usize>) -> Self {
        Indices {
            n_features,
            active_features: active_features.into_iter().collect(),
        }
    }
}

impl Projector for Indices {
    fn n_features(&self) -> usize { self.n_features }

    fn project_ith(&self, _: &[f64], index: IndexT) -> Result<Option<ActivationT>> {
        check_index(index, self.n_features, || {
            Ok(if self.active_features.contains(&index) { Some(1.0) } else { None })
        })
    }

    fn project(&self, _: &[f64]) -> Result<Features> {
        Ok(Features::Sparse(
            self.n_features,
            self.active_features.iter().map(|&i| (i, 1.0)).collect()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    quickcheck! {
        fn test_project_general(constants: Vec<f64>, input: Vec<f64>) -> bool {
            let projector = Constants::new(constants.clone());
            let activations = projector.project(&input).unwrap().expanded().into_raw_vec();

            activations.len() == constants.len() && activations == constants
        }
    }
}
