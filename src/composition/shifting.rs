use crate::{
    basis::Projector,
    core::Features,
    geometry::{Card, Space},
};

/// Shift the output of a `Projector` instance by some fixed amount.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct Shift<P> {
    projector: P,
    offset: f64,
}

impl<P> Shift<P> {
    pub fn new(projector: P, offset: f64) -> Self {
        Shift {
            projector: projector,
            offset: offset,
        }
    }
}

impl<P: Space> Space for Shift<P> {
    type Value = Features;

    fn dim(&self) -> usize { self.projector.dim() }

    fn card(&self) -> Card { self.projector.card() }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for Shift<P> {
    fn project(&self, input: &I) -> Features {
        Features::Dense(self.offset + self.projector.project_expanded(input))
    }
}

#[cfg(test)]
mod tests {
    use crate::basis::fixed::Constant;
    use quickcheck::quickcheck;
    use super::*;

    #[test]
    fn test_shifting() {
        fn prop_output(length: usize, v1: f64, v2: f64) -> bool {
            let p = Shift::new(Constant::new(length, v1), v2);

            p.project_expanded(&[0.0]).into_iter().all(|&v| (v - (v1 + v2)) < 1e-7)
        }

        quickcheck(prop_output as fn(usize, f64, f64) -> bool);
    }
}
