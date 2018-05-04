use geometry::{Space, Card};
use projectors::{Projector, Projection, DenseT};
use rand::ThreadRng;
use std::marker::PhantomData;

fn stack_projections(p1: Projection, n1: usize, p2: Projection, n2: usize) -> Projection {
    use Projection::*;

    match (p1, p2) {
        (Sparse(mut p1_indices), Sparse(p2_indices)) => {
            p2_indices.iter().for_each(|&i| { p1_indices.insert(i+n1); });

            Sparse(p1_indices)
        },
        (p1, p2) => {
            let mut all_activations = p1.expanded(n1).to_vec();
            all_activations.extend_from_slice(p2.expanded(n2).as_slice().unwrap());

            Dense(DenseT::from_vec(all_activations))
        },
    }
}

pub struct Stack<I: ?Sized, P1: Projector<I>, P2: Projector<I>> {
    p1: P1,
    p2: P2,

    phantom: PhantomData<I>,
}

impl<I: ?Sized, P1: Projector<I>, P2: Projector<I>> Stack<I, P1, P2> {
    pub fn new(p1: P1, p2: P2) -> Self {
        Stack {
            p1: p1,
            p2: p2,

            phantom: PhantomData,
        }
    }

}

impl<I: ?Sized, P1: Projector<I>, P2: Projector<I>> Space for Stack<I, P1, P2> {
    type Value = Projection;

    fn sample(&self, rng: &mut ThreadRng) -> Projection {
        let (n1, n2) = (self.p1.dim(), self.p2.dim());

        stack_projections(self.p1.sample(rng), n1, self.p2.sample(rng), n2)
    }

    fn dim(&self) -> usize {
        self.p1.dim() + self.p2.dim()
    }

    fn card(&self) -> Card {
        self.p1.card() * self.p2.card()
    }
}

impl<I: ?Sized, P1: Projector<I>, P2: Projector<I>> Projector<I> for Stack<I, P1, P2> {
    fn project(&self, input: &I) -> Projection {
        let (n1, n2) = (self.p1.dim(), self.p2.dim());

        stack_projections(self.p1.project(input), n1, self.p2.project(input), n2)
    }
}

