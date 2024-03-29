use super::*;
use ndarray::Array1;
use std::ops::MulAssign;

/// Nesterov's Accelerated Gradient descent.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct NAG {
    momentum: f64,
    learning_rate: f64,

    velocity: Array1<f64>,
}

impl NAG {
    pub fn new(n_params: usize, momentum: f64, learning_rate: f64) -> Self {
        NAG {
            momentum,
            learning_rate,
            velocity: Array1::zeros(n_params),
        }
    }
}

impl Optimiser<Features> for NAG {
    fn step_scaled(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
        scale_factor: f64,
    ) -> Result<()> {
        let m = self.momentum;
        let lr = self.learning_rate;

        match features {
            Features::Dense(da) => self
                .velocity
                .zip_mut_with(da, |x, y| *x = m * *x + y * scale_factor),
            Features::Sparse(sa) => {
                self.velocity.mul_assign(m);

                sa.iter().for_each(|(i, a)| {
                    let g = a * scale_factor;

                    self.velocity[*i] += g;
                });
            }
        }
        {
            weights.scaled_add(lr, &self.velocity);
            features.scaled_addto(m, weights);
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.velocity.fill(0.0);
    }
}
