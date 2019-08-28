use crate::{
    Approximator, Parameterised, Features,
    EvaluationResult, UpdateResult,
    Weights, WeightsView, WeightsViewMut,
};

/// `Weights`-`Features` evaluator with vector `Vec<f64>` output.
#[derive(Clone, Debug, PartialEq, Parameterised)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct VectorFunction {
    pub weights: Weights,
}

impl VectorFunction {
    pub fn new(weights: Weights) -> Self {
        VectorFunction { weights, }
    }

    pub fn zeros(n_features: usize, n_outputs: usize) -> Self {
        VectorFunction {
            weights: Weights::zeros((n_features, n_outputs)),
        }
    }
}

impl Approximator for VectorFunction {
    type Output = Vec<f64>;

    fn n_outputs(&self) -> usize { self.weights.cols() }

    fn evaluate(&self, f: &Features) -> EvaluationResult<Self::Output> {
        Ok(f.matmul(&self.weights.view()).into_raw_vec())
    }

    fn update<O: crate::optim::Optimiser>(&mut self, opt: &mut O, f: &Features, es: Vec<f64>) -> UpdateResult<()> {
        es.into_iter()
            .zip(self.weights.gencolumns_mut().into_iter())
            .fold(Ok(()), |acc, (e, mut c)| acc.and(opt.step(&mut c, f, e)))
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
    use super::VectorFunction;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[test]
    fn test_sparse_update_eval() {
        let projector = TileCoding::new(SHBuilder::default(), 4, 100).normalise_l2();

        let mut evaluator = VectorFunction::zeros(projector.n_features(), 2);
        let mut opt = SGD(1.0);

        assert_eq!(evaluator.n_outputs(), 2);
        assert_eq!(evaluator.weights.len(), 200);

        let features = projector.project(&vec![5.0]);

        let _ = evaluator.update_with(&mut opt, &features, vec![20.0, 50.0]);
        let out = evaluator.evaluate(&features).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_dense_update_eval() {
        let projector = Fourier::new(3, vec![(0.0, 10.0)]).normalise_l2();

        let mut fa = VectorFunction::zeros(projector.n_features(), 2);
        let mut opt = SGD(1.0);

        assert_eq!(fa.n_outputs(), 2);
        assert_eq!(fa.weights.len(), 6);

        let features = projector.project(&vec![5.0]);

        let _ = fa.update_with(&mut opt, &features, vec![20.0, 50.0]);
        let out = fa.evaluate(&features).unwrap();

        assert!((out[0] - 20.0).abs() < 1e-6);
        assert!((out[1] - 50.0).abs() < 1e-6);
    }
}
