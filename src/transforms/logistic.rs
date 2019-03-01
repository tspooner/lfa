use super::Transform;

// f(x) ≜ 1 / (1 + exp(-x))
pub struct Logistic;

impl Transform<f64> for Logistic {
    fn transform(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn grad(&self, x: f64) -> f64 {
        let exp_x = x.exp();
        let exp_x_plus_1 = exp_x + 1.0;

        exp_x / exp_x_plus_1 / exp_x_plus_1
    }
}
