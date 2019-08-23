//! Module for SGD-based weight optimisers.
use crate::{Features, UpdateResult};
use ndarray::ArrayViewMut1;

pub trait Optimiser<G = Features> {
    fn step(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &G,
        loss: f64,
    ) -> UpdateResult<()>;

    fn step_batch(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        samples: &[(G, f64)],
    ) -> UpdateResult<()> {
        samples.into_iter().map(|(g, e)| self.step(weights, g, *e)).collect()
    }

    fn reset(&mut self) { unimplemented!() }
}

import_all!(vanilla);
import_all!(implicit);

import_all!(adam);
import_all!(adamax);
import_all!(momentum);
import_all!(nesterov);
