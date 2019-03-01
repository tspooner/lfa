use super::Transform;

// f(x) ≜ exp(x)
pub struct Exp;

impl Transform<f64> for Exp {
    fn transform(&self, x: f64) -> f64 {
        x.exp()
    }

    fn grad(&self, x: f64) -> f64 {
        x * self.transform(x)
    }
}
