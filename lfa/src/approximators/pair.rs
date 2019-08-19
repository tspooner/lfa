use crate::{
    Approximator, Parameterised, Features,
    EvaluationResult, UpdateResult,
    Weights, WeightsView, WeightsViewMut,
};

/// `Weights`-`Features` evaluator with pair `[f64; 2]` output.
#[derive(Clone, Debug, PartialEq, Parameterised)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct PairFunction {
    pub weights: Weights,
}

impl PairFunction {
    pub fn new(weights: Weights) -> Self {
        PairFunction { weights, }
    }

    pub fn zeros(n_features: usize) -> Self {
        PairFunction::new(Weights::zeros((n_features, 2)))
    }
}

impl Approximator for PairFunction {
    type Output = [f64; 2];

    fn n_outputs(&self) -> usize { 2 }

    fn evaluate(&self, features: &Features) -> EvaluationResult<Self::Output> {
        Ok([
            features.dot(&self.weights.column(0)),
            features.dot(&self.weights.column(1)),
        ])
    }

    fn update(&mut self, features: &Features, errors: Self::Output) -> UpdateResult<()> {
        Ok({
            features.scaled_addto(errors[0], &mut self.weights.column_mut(0));
            features.scaled_addto(errors[1], &mut self.weights.column_mut(1));
        })
    }
}

#[cfg(test)]
mod tests {
    extern crate seahash;

    use crate::{
        Approximator,
        basis::{Projector, Fourier, TileCoding},
    };
    use std::hash::BuildHasherDefault;
    use super::PairFunction;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let projector = TileCoding::new(SHBuilder::default(), 4, 100).normalise_l2();
        let mut evaluator = PairFunction::zeros(projector.n_features());

        assert_eq!(evaluator.n_outputs(), 2);
        assert_eq!(evaluator.weights.len(), 200);

        let features = projector.project(&vec![5.0]);

        let _ = evaluator.update(&features, [20.0, 50.0]);
        let out = evaluator.evaluate(&features).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let projector = Fourier::new(3, vec![(0.0, 10.0)]).normalise_l2();
        let mut evaluator = PairFunction::zeros(projector.n_features());

        assert_eq!(evaluator.n_outputs(), 2);
        assert_eq!(evaluator.weights.len(), 6);

        let features = projector.project(&vec![5.0]);

        let _ = evaluator.update(&features, [20.0, 50.0]);
        let out = evaluator.evaluate(&features).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }
}
