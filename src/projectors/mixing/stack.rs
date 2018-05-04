use geometry::{Space, Card};
use projectors::{Projector, Projection, DenseT};
use rand::ThreadRng;
use std::marker::PhantomData;

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

    fn sample(&self, _: &mut ThreadRng) -> Projection {
        unimplemented!()
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
        use Projection::*;

        let p1 = self.p1.project(input);
        let p2 = self.p2.project(input);

        match (p1, p2) {
            (Sparse(mut p1_indices), Sparse(p2_indices)) => {
                let offset: usize = self.p1.dim();

                p2_indices.iter().for_each(|&i| { p1_indices.insert(i+offset); });

                Sparse(p1_indices)
            },
            (p1, p2) => {
                let p1_activations = p1.expanded(self.p1.dim());
                let p2_activations = p2.expanded(self.p2.dim());

                let mut all_activations = p1_activations.to_vec();
                all_activations.extend_from_slice(p2_activations.as_slice().unwrap());

                Dense(DenseT::from_vec(all_activations))
            },
        }
    }
}

