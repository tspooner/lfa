use ndarray::Array1;
use super::*;
use std::ops::MulAssign;

const EPS: f64 = 1e-8;

/// Adaptive moment estimation gradient descent
///
/// https://arxiv.org/pdf/1412.6980.pdf
pub struct Adam {
    beta1: f64,
    beta2: f64,
    learning_rate: f64,

    moment1: Array1<f64>,
    moment2: Array1<f64>,
}

impl Adam {
    pub fn new(n_params: usize, learning_rate: f64, beta1: f64, beta2: f64) -> Self {
        Adam {
            beta1, beta2, learning_rate,

            moment1: Array1::zeros(n_params),
            moment2: Array1::zeros(n_params),
        }
    }
}

impl Optimiser<Features> for Adam {
    fn step(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
        loss: f64
    ) -> UpdateResult<()>
    {
        match features {
            Features::Dense(activations) => {
                let m1 = self.moment1.as_slice_memory_order_mut().unwrap();
                let m2 = self.moment2.as_slice_memory_order_mut().unwrap();

                for (i, a) in activations.indexed_iter() {
                    let g = a * loss;

                    let m1_new = self.beta1 * m1[i] + (1.0 - self.beta1) * g;
                    let m2_new = self.beta2 * m2[i] + (1.0 - self.beta2) * g * g;

                    let m1_unbiased = m1_new / (1.0 - self.beta1);
                    let m2_unbiased = m2_new / (1.0 - self.beta2);

                    m1[i] = m1_new;
                    m2[i] = m2_new;
                    weights[i] +=
                        self.learning_rate * m1_unbiased / (m2_unbiased.sqrt() + EPS);
                }
            },
            Features::Sparse(_, activations) => {
                self.moment1.mul_assign(self.beta1);
                self.moment2.mul_assign(self.beta2);

                let m1 = self.moment1.as_slice_memory_order_mut().unwrap();
                let m2 = self.moment2.as_slice_memory_order_mut().unwrap();

                for (&i, a) in activations.iter() {
                    let g = a * loss;

                    let m1_new = m1[i] + (1.0 - self.beta1) * g;
                    let m2_new = m2[i] + (1.0 - self.beta2) * g * g;

                    let m1_unbiased = m1_new / (1.0 - self.beta1);
                    let m2_unbiased = m2_new / (1.0 - self.beta2);

                    m1[i] = m1_new;
                    m2[i] = m2_new;
                    weights[i] +=
                        self.learning_rate * m1_unbiased / (m2_unbiased.sqrt() + EPS);
                }
            },
        }

        Ok(())
    }
}
