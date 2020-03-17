//! SGD-based weight optimisers.
use crate::{Features, Result};
use ndarray::ArrayViewMut1;

pub trait Optimiser<G = Features> {
    fn step(&mut self, weights: &mut ArrayViewMut1<f64>, grad: &G) -> Result<()> {
        self.step_scaled(weights, grad, 1.0)
    }

    fn step_scaled(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        grad: &G,
        scale_factor: f64
    ) -> Result<()>;

    fn reset(&mut self) {}
}

import_all!(sgd);
import_all!(isgd);

import_all!(momentum);
import_all!(nesterov);

import_all!(adam);
import_all!(adamax);
import_all!(adagrad);
