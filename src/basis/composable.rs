use crate::basis::composition::*;
use crate::geometry::Space;

pub trait Composable: Sized {
    ///
    fn stack<P>(self, p: P) -> Stack<Self, P> { Stack::new(self, p) }

    ///
    fn sum<P: Space>(self, p: P) -> Sum<Self, P>
    where Self: Space {
        Sum::new(self, p)
    }

    ///
    fn shift(self, offset: f64) -> Shift<Self> { Shift::new(self, offset) }

    ///
    fn product<P: Space>(self, p: P) -> Product<Self, P>
    where Self: Space {
        Product::new(self, p)
    }

    ///
    fn scale(self, factor: f64) -> Scale<Self> { Scale::new(self, factor) }

    ///
    fn normalise_l1(self) -> L1Normalise<Self> { L1Normalise::new(self) }

    ///
    fn normalise_l2(self) -> L2Normalise<Self> { L2Normalise::new(self) }

    ///
    fn normalise_lp(self, p: u8) -> LpNormalise<Self> { LpNormalise::new(self, p) }

    ///
    fn normalise_linf(self) -> LinfNormalise<Self> { LinfNormalise::new(self) }
}

#[cfg(test)]
mod tests {
    use super::Composable;
    use crate::basis::{composition::Stack, fixed::Constant, Projector};
    use quickcheck::quickcheck;

    #[test]
    fn test_stack_constant() {
        fn prop_equivalency(input: Vec<f64>) -> bool {
            let p1 = Stack::new(Constant::zeros(10), Constant::ones(10));
            let p2 = Constant::zeros(10).stack(Constant::ones(10));

            p1.project(&input) == p2.project(&input)
        }

        quickcheck(prop_equivalency as fn(Vec<f64>) -> bool);
    }
}
