use crate::{
    basis::Projector,
    core::Features,
    geometry::{Card, Space},
};

/// Scale the output of a `Projector` instance by some fixed amount.
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct Scale<P> {
    projector: P,
    scale: f64,
}

impl<P> Scale<P> {
    pub fn new(projector: P, scale: f64) -> Self {
        Scale {
            projector: projector,
            scale: scale,
        }
    }
}

impl<P: Space> Space for Scale<P> {
    type Value = Features;

    fn dim(&self) -> usize { self.projector.dim() }

    fn card(&self) -> Card { self.projector.card() }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for Scale<P> {
    fn project(&self, input: &I) -> Features {
        Features::Dense(self.scale * self.projector.project_expanded(input))
    }
}

#[cfg(test)]
mod tests {
    use crate::basis::fixed::Constant;
    use quickcheck::quickcheck;
    use super::*;

    #[test]
    fn test_scaling() {
        fn prop_output(length: usize, v1: f64, v2: f64) -> bool {
            let p = Scale::new(Constant::new(length, v1), v2);

            p.project_expanded(&[0.0]).into_iter().all(|&v| (v - v1 * v2) < 1e-7)
        }

        quickcheck(prop_output as fn(usize, f64, f64) -> bool);
    }
}
