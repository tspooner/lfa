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
        scale_factor: f64,
    ) -> Result<()>;

    fn reset(&mut self) {}
}

mod sgd;
pub use self::sgd::SGD;

mod isgd;
pub use self::isgd::ISGD;

mod momentum;
pub use self::momentum::SGDMomentum;

mod nesterov;
pub use self::nesterov::NAG;

mod adam;
pub use self::adam::Adam;

mod adamax;
pub use self::adamax::AdaMax;

mod adagrad;
pub use self::adagrad::Adagrad;
