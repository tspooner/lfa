use crate::geometry::Vector;
use super::Transform;

// f(x) ≜ L / (1 + exp(-k(x - x0))
pub struct Logistic {
    amplitude: f64,
    growth_rate: f64,
    midpoint: f64,
}

impl Logistic {
    pub fn new(amplitude: f64, growth_rate: f64, midpoint: f64) -> Logistic {
        Logistic { amplitude, growth_rate, midpoint, }
    }

    pub fn standard() -> Logistic {
        Logistic::new(1.0, 1.0, 0.0)
    }

    pub fn standard_scaled(amplitude: f64) -> Logistic {
        Logistic::new(amplitude, 1.0, 0.0)
    }

    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn rescale_x(&self, x: f64) -> f64 { self.growth_rate * (x - self.midpoint) }
}

impl Default for Logistic {
    fn default() -> Logistic { Logistic::standard() }
}

impl Transform<f64> for Logistic {
    type Output = f64;

    fn transform(&self, x: f64) -> f64 {
        let x = self.rescale_x(x);

        self.amplitude * Logistic::sigmoid(x)
    }

    fn grad(&self, x: f64) -> f64 {
        let x = self.rescale_x(x);
        let s = Logistic::sigmoid(x);

        self.growth_rate * self.amplitude * (-x).exp() * s * s
    }
}

impl Transform<[f64; 2]> for Logistic {
    type Output = [f64; 2];

    fn transform(&self, x: [f64; 2]) -> [f64; 2] {
        [Transform::<f64>::transform(self, x[0]), Transform::<f64>::transform(self, x[1])]
    }

    fn grad(&self, x: [f64; 2]) -> [f64; 2] {
        [Transform::<f64>::grad(self, x[0]), Transform::<f64>::grad(self, x[1])]
    }
}

impl Transform<[f64; 3]> for Logistic {
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

impl Transform<Vector<f64>> for Logistic {
    type Output = Vector<f64>;

    fn transform(&self, x: Vector<f64>) -> Vector<f64> {
        x.mapv_into(|v| Transform::<f64>::transform(self, v))
    }

    fn grad(&self, x: Vector<f64>) -> Vector<f64> {
        x.mapv_into(|v| Transform::<f64>::grad(self, v))
    }
}

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;
    use std::f64::consts::E;
    use super::{Logistic, Transform};

    #[test]
    fn test_f64() {
        let l = Logistic::standard();

        assert!((l.transform(0.0) - 0.5).abs() < 1e-7);
        assert!((l.transform(1.0) - 1.0 / (1.0 + 1.0 / E)).abs() < 1e-7);
        assert!((l.transform(2.0) - 1.0 / (1.0 + 1.0 / E / E)).abs() < 1e-7);

        assert!((l.grad(0.0) - 0.25).abs() < 1e-5);
        assert!((l.grad(1.0) - 0.196612).abs() < 1e-5);
        assert!((l.grad(2.0) - 0.104994).abs() < 1e-5);
    }

    #[test]
    fn test_f64_positive() {
        fn prop_positive(x: f64) -> bool {
            Logistic::default().transform(x).is_sign_positive()
        }

        quickcheck(prop_positive as fn(f64) -> bool);
    }
}
