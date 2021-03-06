use super::*;
use ndarray::Array1;
use std::ops::MulAssign;

const EPS: f64 = 1e-7;

/// Adaptive moment estimation gradient descent (with infinity norm).
///
/// https://arxiv.org/pdf/1412.6980.pdf
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct AdaMax {
    beta1: f64,
    beta1_prod: f64,

    beta2: f64,

    learning_rate: f64,

    exp_avg: Array1<f64>,
    exp_inf: Array1<f64>,
}

impl AdaMax {
    pub fn new(n_params: usize, learning_rate: f64, beta1: f64, beta2: f64) -> Self {
        AdaMax {
            beta1,
            beta1_prod: beta1,

            beta2,

            learning_rate,

            exp_avg: Array1::zeros(n_params),
            exp_inf: Array1::zeros(n_params),
        }
    }
}

impl Optimiser<Features> for AdaMax {
    fn step_scaled(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
        scale_factor: f64,
    ) -> Result<()>
    {
        self.beta1_prod *= self.beta1;

        match features {
            Features::Dense(da) => {
                let m = self.exp_avg.as_slice_memory_order_mut().unwrap();
                let u = self.exp_inf.as_slice_memory_order_mut().unwrap();

                for (i, a) in da.indexed_iter() {
                    let g = a * scale_factor;

                    let m_new = self.beta1 * m[i] + (1.0 - self.beta1) * g;
                    let u_new = (self.beta2 * u[i]).max(g.abs());
                    let m_unbiased = m_new / (1.0 - self.beta1_prod);

                    m[i] = m_new;
                    u[i] = u_new;
                    weights[i] += self.learning_rate * m_unbiased / (u_new + EPS);
                }
            },
            Features::Sparse(sa) => {
                self.exp_avg.mul_assign(self.beta1);
                self.exp_inf.mul_assign(self.beta2);

                let m = self.exp_avg.as_slice_memory_order_mut().unwrap();
                let u = self.exp_inf.as_slice_memory_order_mut().unwrap();

                for (&i, a) in sa.iter() {
                    let g = a * scale_factor;

                    let m_new = m[i] + (1.0 - self.beta1) * g;
                    let u_new = u[i].max(g.abs());
                    let m_unbiased = m_new / (1.0 - self.beta1_prod);

                    m[i] = m_new;
                    u[i] = u_new;
                    weights[i] += self.learning_rate * m_unbiased / (u_new + EPS);
                }
            },
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.exp_avg.fill(0.0);
        self.exp_inf.fill(0.0);
    }
}
