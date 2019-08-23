use ndarray::Array1;
use super::*;

const EPS: f64 = 1e-8;

pub struct Adagrad {
    learning_rate: f64,

    accumulator: Array1<f64>,
}

impl Adagrad {
    pub fn new(n_params: usize, learning_rate: f64) -> Self {
        Adagrad {
            learning_rate,

            accumulator: Array1::zeros(n_params),
        }
    }
}

impl Optimiser<Features> for Adagrad {
    fn step(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
        loss: f64
    ) -> UpdateResult<()>
    {
        let lr = self.learning_rate;

        match features {
            Features::Dense(activations) => activations.iter()
                .zip(weights.iter_mut())
                .zip(self.accumulator.iter_mut())
                .for_each(|((a, w), ss)| {
                    let g = a * loss;

                    *ss = *ss + g.powi(2);
                    *w = *w + lr * g / (ss.sqrt() + EPS);
                }),
            Features::Sparse(_, activations) => {
                for (i, ss) in self.accumulator.indexed_iter_mut() {
                    let g = activations.get(&i).cloned().unwrap_or(0.0) * loss;

                    *ss = *ss + g.powi(2);
                    weights[i] += lr * g / (ss.sqrt() + EPS);
                }
            },
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.accumulator.fill(0.0);
    }
}
