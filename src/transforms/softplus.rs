use crate::geometry::Vector;
use super::{Transform, Logistic};

// f(x) ≜ log(1 + exp(x))
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct Softplus;

impl Transform<f64> for Softplus {
    type Output = f64;

    fn transform(&self, x: f64) -> f64 {
        (1.0 + x.exp()).ln()
    }

    fn grad(&self, x: f64) -> f64 {
        Logistic::sigmoid(x)
    }
}

impl Transform<[f64; 2]> for Softplus {
    type Output = [f64; 2];

    fn transform(&self, x: [f64; 2]) -> [f64; 2] {
        [Transform::<f64>::transform(self, x[0]), Transform::<f64>::transform(self, x[1])]
    }

    fn grad(&self, x: [f64; 2]) -> [f64; 2] {
        [Transform::<f64>::grad(self, x[0]), Transform::<f64>::grad(self, x[1])]
    }
}

impl Transform<[f64; 3]> for Softplus {
    type Output = [f64; 3];

    fn transform(&self, x: [f64; 3]) -> [f64; 3] {
        [
            Transform::<f64>::transform(self, x[0]),
            Transform::<f64>::transform(self, x[1]),
            Transform::<f64>::transform(self, x[2])
        ]
    }

    fn grad(&self, x: [f64; 3]) -> [f64; 3] {
        [
            Transform::<f64>::grad(self, x[0]),
            Transform::<f64>::grad(self, x[1]),
            Transform::<f64>::grad(self, x[2])
        ]
    }
}

impl Transform<Vector<f64>> for Softplus {
    type Output = Vector<f64>;

    fn transform(&self, x: Vector<f64>) -> Vector<f64> {
        x.mapv_into(|v| Transform::<f64>::transform(self, v))
    }

    fn grad(&self, x: Vector<f64>) -> Vector<f64> {
        x.mapv_into(|v| Transform::<f64>::grad(self, v))
    }
}

// f(x, y, ...) ≜ log(C + exp(x) + exp(y) + ...)
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct LogSumExp(f64);

impl LogSumExp {
    pub fn new(offset: f64) -> LogSumExp { LogSumExp(offset) }
}

impl Default for LogSumExp {
    fn default() -> LogSumExp { LogSumExp::new(0.0) }
}

impl Transform<f64> for LogSumExp {
    type Output = f64;

    fn transform(&self, x: f64) -> f64 {
        (self.0 + x.exp()).ln()
    }

    fn grad(&self, x: f64) -> f64 {
        let exp_term = x.exp();

        exp_term / (self.0 + exp_term)
    }
}

impl Transform<[f64; 2]> for LogSumExp {
    type Output = f64;

    fn transform(&self, x: [f64; 2]) -> f64 {
        (self.0 + x[0].exp() + x[1].exp()).ln()
    }

    fn grad(&self, x: [f64; 2]) -> [f64; 2] {
        let e = [x[0].exp(), x[1].exp()];
        let z = self.0 + e[0] + e[1];

        [e[0] / z, e[1] / z]
    }
}

impl Transform<[f64; 3]> for LogSumExp {
    type Output = f64;

    fn transform(&self, x: [f64; 3]) -> f64 {
        (self.0 + x[0].exp() + x[1].exp() + x[2].exp()).ln()
    }

    fn grad(&self, x: [f64; 3]) -> [f64; 3] {
        let e = [x[0].exp(), x[1].exp(), x[2].exp()];
        let z = self.0 + e[0] + e[1] + e[2];

        [e[0] / z, e[1] / z, e[2] / z]
    }
}

impl Transform<Vector<f64>> for LogSumExp {
    type Output = f64;

    fn transform(&self, x: Vector<f64>) -> f64 {
        (self.0 + x.into_iter().fold(0.0f64, |acc, v| acc + v.exp())).ln()
    }

    fn grad(&self, x: Vector<f64>) -> Vector<f64> {
        let e = x.mapv_into(|v| v.exp());
        let z = self.0 + e.sum();

        e / z
    }
}

#[cfg(test)]
mod tests {
    use crate::transforms::Logistic;
    use quickcheck::quickcheck;
    use super::{Softplus, Transform};

    #[test]
    fn test_f64() {
        assert!((Softplus.transform(0.0) - 0.693147).abs() < 1e-5);
        assert!((Softplus.transform(1.0) - 1.31326).abs() < 1e-5);
        assert!((Softplus.transform(2.0) - 2.12693).abs() < 1e-5);

        assert!((Softplus.grad(0.0) - Logistic::default().transform(0.0)).abs() < 1e-7);
        assert!((Softplus.grad(1.0) - Logistic::default().transform(1.0)).abs() < 1e-7);
        assert!((Softplus.grad(2.0) - Logistic::default().transform(2.0)).abs() < 1e-7);
    }

    #[test]
    fn test_f64_positive() {
        fn prop_positive(x: f64) -> bool {
            Softplus.transform(x).is_sign_positive()
        }

        quickcheck(prop_positive as fn(f64) -> bool);
    }
}
