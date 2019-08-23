use ndarray::Array1;
use super::*;
use std::ops::MulAssign;

/// Nesterov's Accelerated Gradient descent
pub struct NAG {
    momentum: f64,
    learning_rate: f64,

    velocity: Array1<f64>,
    velocity_prev: Array1<f64>,
}

impl NAG {
    pub fn new(n_params: usize, momentum: f64, learning_rate: f64) -> Self {
        NAG {
            momentum, learning_rate,
            velocity: Array1::zeros(n_params),
            velocity_prev: Array1::zeros(n_params),
        }
    }
}

impl Optimiser<Features> for NAG {
    fn step(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
        loss: f64
    ) -> UpdateResult<()> {
        let momentum = self.momentum;
        let learning_rate = self.learning_rate;

        ::std::mem::swap(&mut self.velocity, &mut self.velocity_prev);

        match features {
            Features::Dense(activations) => self.velocity.zip_mut_with(activations, |x, y| {
                *x = momentum * *x + learning_rate * y * loss
            }),
            Features::Sparse(_, activations) => {
                self.velocity.mul_assign(momentum);

                for (i, a) in activations.iter() {
                    self.velocity[*i] += learning_rate * a * loss;
                }
            },
        }

        Ok({
            weights.scaled_add(momentum, &self.velocity);
            features.scaled_addto(learning_rate, weights);
        })
    }
}
