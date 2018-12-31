use crate::basis::{Composable, Projection, Projector};
use crate::geometry::{Card, Space, norms::{l1, l2, lp, linf}};

/// Apply _L₁_ normalisation to the output of a `Projector` instance.
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct L1Normalise<P>(P);

impl<P> L1Normalise<P> {
    pub fn new(projector: P) -> Self {
        L1Normalise(projector)
    }
}

impl<P: Space> Space for L1Normalise<P> {
    type Value = Projection;

    fn dim(&self) -> usize { self.0.dim() }

    fn card(&self) -> Card { self.0.card() }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for L1Normalise<P> {
    fn project(&self, input: &I) -> Projection {
        let phi = self.0.project_expanded(input);
        let z = l1(phi.as_slice().unwrap());

        Projection::Dense(phi / z)
    }
}

impl<P> Composable for L1Normalise<P> {}

/// Apply _L₂_ normalisation to the output of a `Projector` instance.
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct L2Normalise<P>(P);

impl<P> L2Normalise<P> {
    pub fn new(projector: P) -> Self {
        L2Normalise(projector)
    }
}

impl<P: Space> Space for L2Normalise<P> {
    type Value = Projection;

    fn dim(&self) -> usize { self.0.dim() }

    fn card(&self) -> Card { self.0.card() }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for L2Normalise<P> {
    fn project(&self, input: &I) -> Projection {
        let phi = self.0.project_expanded(input);
        let z = l2(phi.as_slice().unwrap());

        Projection::Dense(phi / z)
    }
}

impl<P> Composable for L2Normalise<P> {}

/// Apply _Lp_ normalisation to the output of a `Projector` instance.
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct LpNormalise<P>(P, u8);

impl<P> LpNormalise<P> {
    pub fn new(projector: P, p: u8) -> Self {
        LpNormalise(projector, p)
    }
}

impl<P: Space> Space for LpNormalise<P> {
    type Value = Projection;

    fn dim(&self) -> usize { self.0.dim() }

    fn card(&self) -> Card { self.0.card() }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for LpNormalise<P> {
    fn project(&self, input: &I) -> Projection {
        let phi = self.0.project_expanded(input);
        let z = lp(phi.as_slice().unwrap(), self.1);

        Projection::Dense(phi / z)
    }
}

impl<P> Composable for LpNormalise<P> {}

/// Apply _L∞_ normalisation to the output of a `Projector` instance.
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct LinfNormalise<P>(P);

impl<P> LinfNormalise<P> {
    pub fn new(projector: P) -> Self {
        LinfNormalise(projector)
    }
}

impl<P: Space> Space for LinfNormalise<P> {
    type Value = Projection;

    fn dim(&self) -> usize { self.0.dim() }

    fn card(&self) -> Card { self.0.card() }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for LinfNormalise<P> {
    fn project(&self, input: &I) -> Projection {
        let phi = self.0.project_expanded(input);
        let z = linf(phi.as_slice().unwrap());

        Projection::Dense(phi / z)
    }
}

impl<P> Composable for LinfNormalise<P> {}

#[cfg(test)]
mod tests {
    use crate::basis::fixed::Constant;
    use quickcheck::quickcheck;
    use super::*;

    #[test]
    fn test_l1() {
        fn prop_output(length: usize, value: f64) -> bool {
            let p = L1Normalise::new(Constant::new(length, value));
            let z = length as f64;

            p.project_expanded(&[0.0]).into_iter().all(|&v| (v - 1.0 / z) < 1e-7)
        }

        quickcheck(prop_output as fn(usize, f64) -> bool);
    }

    #[test]
    fn test_l2() {
        fn prop_output(length: usize, value: f64) -> bool {
            let p = L2Normalise::new(Constant::new(length, value));
            let z = (length as f64).sqrt();

            p.project_expanded(&[0.0]).into_iter().all(|&v| (v - 1.0 / z) < 1e-7)
        }

        quickcheck(prop_output as fn(usize, f64) -> bool);
    }

    #[test]
    fn test_lp() {
        fn prop_output(length: usize, value: f64, pow: u8) -> bool {
            let p = LpNormalise::new(Constant::new(length, value), pow);
            let z = (length as f64).powf(1.0 / pow as f64);

            p.project_expanded(&[0.0]).into_iter().all(|&v| (v - 1.0 / z) < 1e-7)
        }

        quickcheck(prop_output as fn(usize, f64, u8) -> bool);
    }

    #[test]
    fn test_linf() {
        fn prop_output(length: usize, value: f64) -> bool {
            let p = LinfNormalise::new(Constant::new(length, value));
            let z = 1.0;

            p.project_expanded(&[0.0]).into_iter().all(|&v| (v - 1.0 / z) < 1e-7)
        }

        quickcheck(prop_output as fn(usize, f64) -> bool);
    }
}
