use ndarray::Array1;
use super::*;
use std::ops::{AddAssign, MulAssign};

pub struct SGDMomentum {
    momentum: f64,
    learning_rate: f64,

    velocity: Array1<f64>,
}

impl SGDMomentum {
    pub fn new(n_params: usize, momentum: f64, learning_rate: f64) -> Self {
        SGDMomentum {
            momentum, learning_rate,
            velocity: Array1::zeros(n_params),
        }
    }
}

impl Optimiser<Features> for SGDMomentum {
    fn step(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
        loss: f64
    ) -> UpdateResult<()> {
        let m = self.momentum;
        let lr = self.learning_rate;

        match features {
            Features::Dense(activations) => self.velocity.zip_mut_with(activations, |x, y| {
                *x = m * *x + lr * y * loss
            }),
            Features::Sparse(_, activations) => {
                self.velocity.mul_assign(m);

                for (i, a) in activations.iter() {
                    self.velocity[*i] += lr * a * loss;
                }
            },
        }

        Ok(weights.add_assign(&self.velocity))
    }

    fn reset(&mut self) {
        self.velocity.fill(0.0);
    }
}
