// TODO: ANOVA kernel

pub trait Kernel<I> {
    fn kernel(&self, x: I, y: I) -> f64;
}

mod stationary;
pub use self::stationary::*;

mod rational_quadratic;
pub use self::rational_quadratic::*;
