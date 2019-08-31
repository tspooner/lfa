use crate::{Approximator, Parameterised, Features, Result, WeightsView, WeightsViewMut};
use ndarray::Array1;

/// [`Weights`]-[`Features`] evaluator with `f64` output.
///
/// [`Weights`]: type.Weights.html
/// [`Features`]: enum.Features.html
#[derive(Clone, Debug, PartialEq, Parameterised)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ScalarFunction {
    /// Approximation weights.
    pub weights: Array1<f64>,
}

impl ScalarFunction {
    /// Construct a new approximation with specified `weights`.
    pub fn new(weights: Array1<f64>) -> Self {
        ScalarFunction { weights, }
    }

    /// Construct a new approximation with zeroed weights.
    pub fn zeros(n_features: usize) -> Self {
        ScalarFunction::new(Array1::zeros((n_features,)))
    }
}

impl Approximator for ScalarFunction {
    type Output = f64;

    fn n_outputs(&self) -> usize { 1 }

    fn evaluate(&self, features: &Features) -> Result<Self::Output> {
        Ok(features.dot(&self.weights.view()))
    }

    fn update<O>(&mut self, opt: &mut O, f: &Features, e: f64) -> Result<()>
    where
        O: crate::optim::Optimiser,
    {
        opt.step(&mut self.weights.view_mut(), f, e)
    }
}

#[cfg(test)]
mod tests {
    extern crate seahash;

    use crate::{
        Approximator,
        basis::{Projector, Fourier, TileCoding},
        optim::SGD,
    };
    use std::hash::BuildHasherDefault;
    use super::ScalarFunction;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let projector = TileCoding::new(SHBuilder::default(), 4, 100).normalise_l2();

        let mut fa = ScalarFunction::zeros(projector.n_features());
        let mut opt = SGD(1.0);

        assert_eq!(fa.n_outputs(), 1);
        assert_eq!(fa.weights.len(), 100);

        let features = projector.project(&vec![5.0]).unwrap();

        let _ = fa.update(&mut opt, &features, 50.0);
        let out = fa.evaluate(&features).unwrap();

        assert!((out - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let projector = Fourier::new(3, vec![(0.0, 10.0)]).normalise_l2();

        let mut fa = ScalarFunction::zeros(projector.n_features());
        let mut opt = SGD(1.0);

        assert_eq!(fa.n_outputs(), 1);
        assert_eq!(fa.weights.len(), 3);

        let features = projector.project(&vec![5.0]).unwrap();

        let _ = fa.update(&mut opt, &features, 50.0);
        let out = fa.evaluate(&features).unwrap();

        assert!((out - 50.0).abs() < 1e-6);
    }
}
