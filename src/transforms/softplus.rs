use super::{Transform, Logistic};

// f(x) ≜ log(1 + exp(x))
pub struct Softplus;

impl Transform<f64> for Softplus {
    fn transform(&self, x: f64) -> f64 {
        (1.0 + x.exp()).ln()
    }

    fn grad(&self, x: f64) -> f64 {
        Logistic.transform(x)
    }
}
