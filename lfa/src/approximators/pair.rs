use crate::{Approximator, Parameterised, Features, Result, Weights, WeightsView, WeightsViewMut};

/// [`Weights`]-[`Features`] evaluator with `[f64; 2]` output.
///
/// [`Weights`]: type.Weights.html
/// [`Features`]: enum.Features.html
#[derive(Clone, Debug, PartialEq, Parameterised)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct PairFunction {
    /// Approximation weights.
    pub weights: Weights,
}

impl PairFunction {
    /// Construct a new approximation with specified `weights`.
    pub fn new(weights: Weights) -> Self {
        PairFunction { weights, }
    }

    /// Construct a new approximation with zeroed weights.
    pub fn zeros(n_features: usize) -> Self {
        PairFunction::new(Weights::zeros((n_features, 2)))
    }
}

impl Approximator for PairFunction {
    type Output = [f64; 2];

    fn n_outputs(&self) -> usize { 2 }

    fn evaluate(&self, features: &Features) -> Result<Self::Output> {
        Ok([
            features.dot(&self.weights.column(0)),
            features.dot(&self.weights.column(1)),
        ])
    }

    fn update<O>(&mut self, opt: &mut O, f: &Features, es: [f64; 2]) -> Result<()>
    where
        O: crate::optim::Optimiser,
    {
        opt.step(&mut self.weights.column_mut(0), f, es[0])
            .and(opt.step(&mut self.weights.column_mut(1), f, es[1]))
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
    use super::PairFunction;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let projector = TileCoding::new(SHBuilder::default(), 4, 100).normalise_l2();

        let mut fa = PairFunction::zeros(projector.n_features());
        let mut opt = SGD(1.0);

        assert_eq!(fa.n_outputs(), 2);
        assert_eq!(fa.weights.len(), 200);

        let features = projector.project(&vec![5.0]).unwrap();

        let _ = fa.update(&mut opt, &features, [20.0, 50.0]);
        let out = fa.evaluate(&features).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let projector = Fourier::new(3, vec![(0.0, 10.0)]).normalise_l2();

        let mut fa = PairFunction::zeros(projector.n_features());
        let mut opt = SGD(1.0);

        assert_eq!(fa.n_outputs(), 2);
        assert_eq!(fa.weights.len(), 6);

        let features = projector.project(&vec![5.0]).unwrap();

        let _ = fa.update(&mut opt, &features, [20.0, 50.0]);
        let out = fa.evaluate(&features).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }
}
