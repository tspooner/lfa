use geometry::Vector;

pub trait Gradient {
    fn chain(self, other: Self) -> Self;
}

impl Gradient for f64 {
    fn chain(self, other: f64) -> f64 {
        self * other
    }
}

impl Gradient for (f64, f64) {
    fn chain(self, other: (f64, f64)) -> (f64, f64) {
        (self.0 * other.0, self.1 * other.1)
    }
}

impl Gradient for (f64, f64, f64) {
    fn chain(self, other: (f64, f64, f64)) -> (f64, f64, f64) {
        (self.0 * other.0, self.1 * other.1, self.2 * other.2)
    }
}

impl Gradient for Vector<f64> {
    fn chain(self, other: Vector<f64>) -> Vector<f64> {
        self * other
    }
}
