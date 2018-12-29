use crate::basis::composition::{Sum, Product, Stack, Shift, Scale};
use crate::geometry::Space;

pub trait Composable {
    ///
    fn stack<P>(self, p: P) -> Stack<Self, P> where Self: Sized {
        Stack::new(self, p)
    }

    ///
    fn sum<P: Space>(self, p: P) -> Sum<Self, P> where Self: Sized + Space {
        Sum::new(self, p)
    }

    ///
    fn shift(self, offset: f64) -> Shift<Self> where Self: Sized {
        Shift::new(self, offset)
    }

    ///
    fn product<P: Space>(self, p: P) -> Product<Self, P> where Self: Sized + Space {
        Product::new(self, p)
    }

    ///
    fn scale(self, factor: f64) -> Scale<Self> where Self: Sized {
        Scale::new(self, factor)
    }
}

#[cfg(test)]
mod tests {
    use crate::basis::{Projector, composition::Stack, fixed::Constant};
    use quickcheck::quickcheck;
    use super::Composable;

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
