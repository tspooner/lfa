use ndarray::Array1;
use super::*;
use std::ops::MulAssign;

/// Nesterov's Accelerated Gradient descent
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct NAG {
    momentum: f64,
    learning_rate: f64,

    velocity: Array1<f64>,
}

impl NAG {
    pub fn new(n_params: usize, momentum: f64, learning_rate: f64) -> Self {
        NAG {
            momentum, learning_rate,
            velocity: Array1::zeros(n_params),
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
        let m = self.momentum;
        let lr = self.learning_rate;

        match features {
            Features::Dense(activations) => self.velocity.zip_mut_with(activations, |x, y| {
                let g = y * loss;

                *x = m * *x + g
            }),
            Features::Sparse(_, activations) => {
                self.velocity.mul_assign(m);

                for (i, a) in activations.iter() {
                    let g = a * loss;

                    self.velocity[*i] += g;
                }
            },
        }

        Ok({
            weights.scaled_add(lr, &self.velocity);
            features.scaled_addto(m * loss, weights);
        })
    }

    fn reset(&mut self) {
        self.velocity.fill(0.0);
    }
}
