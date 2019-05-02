use crate::{
    basis::Projector,
    core::Features,
    geometry::{Card, Space},
};

/// Stack the output of two `Projector` instances.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct Stack<P1, P2> {
    p1: P1,
    p2: P2,
}

impl<P1, P2> Stack<P1, P2> {
    pub fn new(p1: P1, p2: P2) -> Self { Stack { p1, p2 } }
}

impl<P1: Space, P2: Space> Space for Stack<P1, P2> {
    type Value = Features;

    fn dim(&self) -> usize { self.p1.dim() + self.p2.dim() }

    fn card(&self) -> Card { self.p1.card() * self.p2.card() }
}

impl<I: ?Sized, P1: Projector<I>, P2: Projector<I>> Projector<I> for Stack<P1, P2> {
    fn project(&self, input: &I) -> Features {
        self.p1.project(input).stack(self.p1.dim(), self.p2.project(input), self.p2.dim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{
        fixed::{Constant, Indices},
        Projector,
    };
    use crate::geometry::Vector;
    use std::iter;

    #[test]
    fn test_stack_constant() {
        let p = Stack::new(Constant::zeros(10), Constant::ones(10));
        let output: Features =
            Vector::from_iter(iter::repeat(0.0).take(10).chain(iter::repeat(1.0).take(10))).into();

        assert_eq!(p.dim(), 20);
        assert_eq!(p.project(&[0.0]), output);
        assert_eq!(p.project(&[0.0, 1.0]), output);
        assert_eq!(p.project(&[-1.0, 1.0]), output);
    }

    #[test]
    fn test_stack_indices() {
        let p = Stack::new(Indices::new(10, vec![5]), Indices::new(10, vec![0]));
        let output: Features = vec![5, 10].into();

        assert_eq!(p.dim(), 20);
        assert_eq!(p.project(&[0.0]), output);
        assert_eq!(p.project(&[0.0, 1.0]), output);
        assert_eq!(p.project(&[-1.0, 1.0]), output);
    }

    #[test]
    fn test_stack_mixed() {
        let p = Stack::new(Constant::ones(10), Indices::new(10, vec![0]));
        let output: Features =
            Vector::from_iter(iter::repeat(1.0).take(11).chain(iter::repeat(0.0).take(9))).into();

        assert_eq!(p.dim(), 20);
        assert_eq!(p.project(&[0.0]), output);
        assert_eq!(p.project(&[0.0, 1.0]), output);
        assert_eq!(p.project(&[-1.0, 1.0]), output);
    }
}
