use crate::{
    Approximator, Parameterised, Features,
    EvaluationResult, UpdateResult,
    Weights, WeightsView, WeightsViewMut,
};

/// `Weights`-`Features` evaluator with triple `[f64; 3]` output.
#[derive(Clone, Debug, PartialEq, Parameterised)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TripleFunction {
    pub weights: Weights,
}

impl TripleFunction {
    pub fn new(weights: Weights) -> Self {
        TripleFunction { weights, }
    }

    pub fn zeros(n_features: usize) -> Self {
        TripleFunction::new(Weights::zeros((n_features, 3)))
    }
}

impl Approximator for TripleFunction {
    type Output = [f64; 3];

    fn n_outputs(&self) -> usize { 3 }

    fn evaluate(&self, features: &Features) -> EvaluationResult<Self::Output> {
        Ok([
            features.dot(&self.weights.column(0)),
            features.dot(&self.weights.column(1)),
            features.dot(&self.weights.column(2)),
        ])
    }

    fn update_with<O: crate::sgd::Optimiser>(&mut self, opt: &mut O, f: &Features, es: [f64; 3]) -> UpdateResult<()> {
        opt.step(&mut self.weights.column_mut(0), f, es[0])
            .and(opt.step(&mut self.weights.column_mut(1), f, es[1]))
            .and(opt.step(&mut self.weights.column_mut(2), f, es[2]))
    }
}

#[cfg(test)]
mod tests {
    extern crate seahash;

    use crate::{
        Approximator,
        basis::{Projector, Fourier, TileCoding},
        sgd::SGD,
    };
    use std::hash::BuildHasherDefault;
    use super::TripleFunction;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let projector = TileCoding::new(SHBuilder::default(), 4, 100).normalise_l2();

        let mut fa = TripleFunction::zeros(projector.n_features());
        let mut opt = SGD(1.0);

        assert_eq!(fa.n_outputs(), 3);
        assert_eq!(fa.weights.len(), 300);

        let features = projector.project(&vec![5.0]);

        let _ = fa.update_with(&mut opt, &features, [20.0, 50.0, 100.0]);
        let out = fa.evaluate(&features).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
        assert!((out[2] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let projector = Fourier::new(3, vec![(0.0, 10.0)]).normalise_l2();

        let mut fa = TripleFunction::zeros(projector.n_features());
        let mut opt = SGD(1.0);

        assert_eq!(fa.n_outputs(), 3);
        assert_eq!(fa.weights.len(), 9);

        let features = projector.project(&vec![5.0]);

        let _ = fa.update_with(&mut opt, &features, [20.0, 50.0, 100.0]);
        let out = fa.evaluate(&features).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
        assert!((out[2] - 100.0).abs() < 1e-6);
    }
}
