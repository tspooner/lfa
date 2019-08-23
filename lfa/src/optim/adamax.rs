use ndarray::Array1;
use super::*;
use std::ops::MulAssign;

const EPS: f64 = 1e-8;

/// Adaptive moment estimation gradient descent (with infinity norm)
///
/// https://arxiv.org/pdf/1412.6980.pdf
pub struct AdaMax {
    beta1: f64,
    beta2: f64,
    learning_rate: f64,

    moment: Array1<f64>,
    inf_norm: Array1<f64>,
}

impl AdaMax {
    pub fn new(n_params: usize, learning_rate: f64, beta1: f64, beta2: f64) -> Self {
        AdaMax {
            beta1, beta2, learning_rate,

            moment: Array1::zeros(n_params),
            inf_norm: Array1::zeros(n_params),
        }
    }
}

impl Optimiser<Features> for AdaMax {
    fn step(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
        loss: f64
    ) -> UpdateResult<()>
    {
        match features {
            Features::Dense(activations) => {
                let m = self.moment.as_slice_memory_order_mut().unwrap();
                let u = self.inf_norm.as_slice_memory_order_mut().unwrap();

                for (i, a) in activations.indexed_iter() {
                    let g = a * loss;

                    let m_new = self.beta1 * m[i] + (1.0 - self.beta1) * g;
                    let u_new = (self.beta2 * u[i]).max(g.abs());
                    let m_unbiased = m_new / (1.0 - self.beta1);

                    m[i] = m_new;
                    u[i] = u_new;
                    weights[i] += self.learning_rate * m_unbiased / (u_new + EPS);
                }
            },
            Features::Sparse(_, activations) => {
                self.moment.mul_assign(self.beta1);
                self.inf_norm.mul_assign(self.beta2);

                let m = self.moment.as_slice_memory_order_mut().unwrap();
                let u = self.inf_norm.as_slice_memory_order_mut().unwrap();

                for (&i, a) in activations.iter() {
                    let g = a * loss;

                    let m_new = m[i] + (1.0 - self.beta1) * g;
                    let u_new = u[i].max(g.abs());
                    let m_unbiased = m_new / (1.0 - self.beta1);

                    m[i] = m_new;
                    u[i] = u_new;
                    weights[i] += self.learning_rate * m_unbiased / (u_new + EPS);
                }
            },
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.moment.fill(0.0);
        self.inf_norm.fill(0.0);
    }
}
