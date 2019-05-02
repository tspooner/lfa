use crate::geometry::Vector;
use super::Transform;

// f(x) ≜ tanh(x)
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct Tanh;

impl Transform<f64> for Tanh {
    type Output = f64;

    fn transform(&self, x: f64) -> f64 {
        x.tanh()
    }

    fn grad(&self, x: f64) -> f64 {
        let cosh = x.cosh();

        1.0 / cosh / cosh
    }
}

impl Transform<[f64; 2]> for Tanh {
    type Output = [f64; 2];

    fn transform(&self, x: [f64; 2]) -> [f64; 2] {
        [Transform::<f64>::transform(self, x[0]), Transform::<f64>::transform(self, x[1])]
    }

    fn grad(&self, x: [f64; 2]) -> [f64; 2] {
        [Transform::<f64>::grad(self, x[0]), Transform::<f64>::grad(self, x[1])]
    }
}

impl Transform<[f64; 3]> for Tanh {
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

impl Transform<Vector<f64>> for Tanh {
    type Output = Vector<f64>;

    fn transform(&self, x: Vector<f64>) -> Vector<f64> {
        x.mapv_into(|v| Transform::<f64>::transform(self, v))
    }

    fn grad(&self, x: Vector<f64>) -> Vector<f64> {
        x.mapv_into(|v| Transform::<f64>::grad(self, v))
    }
}
