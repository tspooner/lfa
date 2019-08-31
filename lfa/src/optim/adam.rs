use ndarray::Array1;
use super::*;
use std::ops::MulAssign;

const EPS: f64 = 1e-7;

/// Adaptive moment estimation gradient descent
///
/// https://arxiv.org/pdf/1412.6980.pdf
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Adam {
    beta1: f64,
    beta1_prod: f64,

    beta2: f64,
    beta2_prod: f64,

    learning_rate: f64,

    exp_avg: Array1<f64>,
    exp_avg_sq: Array1<f64>,
}

impl Adam {
    pub fn new(n_params: usize, learning_rate: f64, beta1: f64, beta2: f64) -> Self {
        Adam {
            beta1,
            beta1_prod: beta1,

            beta2,
            beta2_prod: beta2,

            learning_rate,

            exp_avg: Array1::zeros(n_params),
            exp_avg_sq: Array1::zeros(n_params),
        }
    }
}

impl Optimiser<Features> for Adam {
    fn step(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
        loss: f64
    ) -> Result<()>
    {
        self.beta1_prod *= self.beta1;
        self.beta2_prod *= self.beta2;

        match features {
            Features::Dense(activations) => {
                let m1 = self.exp_avg.as_slice_memory_order_mut().unwrap();
                let m2 = self.exp_avg_sq.as_slice_memory_order_mut().unwrap();

                for (i, a) in activations.indexed_iter() {
                    let g = a * loss;

                    let m1_new = self.beta1 * m1[i] + (1.0 - self.beta1) * g;
                    let m2_new = self.beta2 * m2[i] + (1.0 - self.beta2) * g * g;

                    let m1_unbiased = m1_new / (1.0 - self.beta1_prod);
                    let m2_unbiased = m2_new / (1.0 - self.beta2_prod);

                    m1[i] = m1_new;
                    m2[i] = m2_new;
                    weights[i] += self.learning_rate * m1_unbiased / (m2_unbiased.sqrt() + EPS);
                }
            },
            Features::Sparse(_, activations) => {
                self.exp_avg.mul_assign(self.beta1);
                self.exp_avg_sq.mul_assign(self.beta2);

                let m1 = self.exp_avg.as_slice_memory_order_mut().unwrap();
                let m2 = self.exp_avg_sq.as_slice_memory_order_mut().unwrap();

                for (&i, a) in activations.iter() {
                    let g = a * loss;

                    let m1_new = m1[i] + (1.0 - self.beta1) * g;
                    let m2_new = m2[i] + (1.0 - self.beta2) * g * g;

                    let m1_unbiased = m1_new / (1.0 - self.beta1_prod);
                    let m2_unbiased = m2_new / (1.0 - self.beta2_prod);

                    m1[i] = m1_new;
                    m2[i] = m2_new;
                    weights[i] += self.learning_rate * m1_unbiased / (m2_unbiased.sqrt() + EPS);
                }
            },
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.exp_avg.fill(0.0);
        self.exp_avg_sq.fill(0.0);
    }
}
