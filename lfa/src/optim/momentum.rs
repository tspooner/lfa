use ndarray::Array1;
use super::*;
use std::ops::MulAssign;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
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
    fn step_scaled(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
        scale_factor: f64
    ) -> Result<()>
    {
        let m = self.momentum;
        let lr = self.learning_rate;

        match features {
            Features::Dense(activations) => self.velocity.zip_mut_with(activations, |x, y| {
                *x = m * *x + y * scale_factor
            }),
            Features::Sparse(_, activations) => {
                self.velocity.mul_assign(m);

                activations.iter().for_each(|(i, a)| {
                    self.velocity[*i] += a * scale_factor;
                });
            },
        }

        Ok(weights.scaled_add(lr, &self.velocity))
    }

    fn reset(&mut self) {
        self.velocity.fill(0.0);
    }
}
