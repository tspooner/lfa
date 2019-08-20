use ndarray::Array1;
use super::*;
use std::ops::{AddAssign, MulAssign};

pub struct SGDM {
    momentum: f64,
    learning_rate: f64,

    velocity: Array1<f64>,
}

impl SGDM {
    pub fn new(n_params: usize, momentum: f64, learning_rate: f64) -> Self {
        SGDM {
            momentum, learning_rate,
            velocity: Array1::zeros(n_params),
        }
    }
}

impl Optimiser<Features> for SGDM {
    fn step(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
        error: f64
    ) -> UpdateResult<()> {
        let momentum = self.momentum;
        let learning_rate = self.learning_rate;

        match features {
            Features::Dense(activations) => self.velocity.zip_mut_with(activations, |x, y| {
                *x = momentum * *x + learning_rate * y * error
            }),
            Features::Sparse(_, activations) => {
                self.velocity.mul_assign(momentum);

                for (i, a) in activations.iter() {
                    self.velocity[*i] += learning_rate * a * error;
                }
            },
        }

        Ok(weights.add_assign(&self.velocity))
    }
}
