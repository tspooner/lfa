use crate::{
    basis::Composable,
    core::{DenseT, Projection, Projector},
    geometry::{Card, Space},
};

fn stack_projections(p1: Projection, n1: usize, p2: Projection, n2: usize) -> Projection {
    match (p1, p2) {
        (Projection::Sparse(mut p1_indices), Projection::Sparse(p2_indices)) => {
            p2_indices.iter().for_each(|&i| {
                p1_indices.insert(i + n1);
            });

            Projection::Sparse(p1_indices)
        },
        (p1, p2) => {
            let mut all_activations = p1.expanded(n1).to_vec();
            all_activations.extend_from_slice(p2.expanded(n2).as_slice().unwrap());

            Projection::Dense(DenseT::from_vec(all_activations))
        },
    }
}

/// Stack the output of two `Projector` instances.
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct Stack<P1, P2> {
    p1: P1,
    p2: P2,
}

impl<P1, P2> Stack<P1, P2> {
    pub fn new(p1: P1, p2: P2) -> Self { Stack { p1, p2 } }
}

impl<P1: Space, P2: Space> Space for Stack<P1, P2> {
    type Value = Projection;

    fn dim(&self) -> usize { self.p1.dim() + self.p2.dim() }

    fn card(&self) -> Card { self.p1.card() * self.p2.card() }
}

impl<I: ?Sized, P1: Projector<I>, P2: Projector<I>> Projector<I> for Stack<P1, P2> {
    fn project(&self, input: &I) -> Projection {
        stack_projections(
            self.p1.project(input),
            self.p1.dim(),
            self.p2.project(input),
            self.p2.dim(),
        )
    }
}

impl<P1, P2> Composable for Stack<P1, P2> {}

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
        let output: Projection =
            Vector::from_iter(iter::repeat(0.0).take(10).chain(iter::repeat(1.0).take(10))).into();

        assert_eq!(p.dim(), 20);
        assert_eq!(p.project(&[0.0]), output);
        assert_eq!(p.project(&[0.0, 1.0]), output);
        assert_eq!(p.project(&[-1.0, 1.0]), output);
    }

    #[test]
    fn test_stack_indices() {
        let p = Stack::new(Indices::new(10, vec![5]), Indices::new(10, vec![0]));
        let output: Projection = vec![5, 10].into();

        assert_eq!(p.dim(), 20);
        assert_eq!(p.project(&[0.0]), output);
        assert_eq!(p.project(&[0.0, 1.0]), output);
        assert_eq!(p.project(&[-1.0, 1.0]), output);
    }

    #[test]
    fn test_stack_mixed() {
        let p = Stack::new(Constant::ones(10), Indices::new(10, vec![0]));
        let output: Projection =
            Vector::from_iter(iter::repeat(1.0).take(11).chain(iter::repeat(0.0).take(9))).into();

        assert_eq!(p.dim(), 20);
        assert_eq!(p.project(&[0.0]), output);
        assert_eq!(p.project(&[0.0, 1.0]), output);
        assert_eq!(p.project(&[-1.0, 1.0]), output);
    }
}
