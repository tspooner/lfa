use geometry::{Space, Card, norms::l1};
use projectors::{Projector, Projection};
use rand::ThreadRng;
use std::marker::PhantomData;

pub struct Sum<I: ?Sized, P1: Projector<I>, P2: Projector<I>> {
    p1: P1,
    p2: P2,

    phantom: PhantomData<I>,
}

impl<I: ?Sized, P1: Projector<I>, P2: Projector<I>> Sum<I, P1, P2> {
    pub fn new(p1: P1, p2: P2) -> Self {
        if p1.dim() != p2.dim() {
            panic!("Projectors p1 and p2 must have the same dimensionality.");
        }

        Sum {
            p1: p1,
            p2: p2,

            phantom: PhantomData,
        }
    }
}

impl<I: ?Sized, P1: Projector<I>, P2: Projector<I>> Space for Sum<I, P1, P2> {
    type Value = Projection;

    fn sample(&self, _: &mut ThreadRng) -> Projection {
        unimplemented!()
    }

    fn dim(&self) -> usize {
        self.p1.dim().max(self.p2.dim())
    }

    fn card(&self) -> Card {
        self.p1.card() * self.p2.card()
    }
}

impl<I: ?Sized, P1: Projector<I>, P2: Projector<I>> Projector<I> for Sum<I, P1, P2> {
    fn project(&self, input: &I) -> Projection {
        use Projection::*;

        let p1 = self.p1.project(input);
        let p2 = self.p2.project(input);

        match (p1, p2) {
            (Sparse(p1_indices), Sparse(p2_indices)) => {
                Sparse(p1_indices.union(&p2_indices).cloned().collect())
            },
            (p1, p2) => {
                let p1_activations = p1.expanded(self.p1.dim());
                let p2_activations = p2.expanded(self.p2.dim());

                Dense(p1_activations + p2_activations)
            },
        }
    }
}

pub struct Product<I: ?Sized, P1: Projector<I>, P2: Projector<I>> {
    p1: P1,
    p2: P2,

    phantom: PhantomData<I>,
}

impl<I: ?Sized, P1: Projector<I>, P2: Projector<I>> Product<I, P1, P2> {
    pub fn new(p1: P1, p2: P2) -> Self {
        if p1.dim() != p2.dim() {
            panic!("Projectors p1 and p2 must have the same dimensionality.");
        }

        Product {
            p1: p1,
            p2: p2,

            phantom: PhantomData,
        }
    }
}

impl<I: ?Sized, P1: Projector<I>, P2: Projector<I>> Space for Product<I, P1, P2> {
    type Value = Projection;

    fn sample(&self, _: &mut ThreadRng) -> Projection {
        unimplemented!()
    }

    fn dim(&self) -> usize {
        self.p1.dim()
    }

    fn card(&self) -> Card { unimplemented!() }
}

impl<I: ?Sized, P1: Projector<I>, P2: Projector<I>> Projector<I> for Product<I, P1, P2> {
    fn project(&self, input: &I) -> Projection {
        use Projection::*;

        let p1 = self.p1.project(input);
        let p2 = self.p2.project(input);

        match (p1, p2) {
            (Sparse(p1_indices), Sparse(p2_indices)) => {
                Sparse(p1_indices.intersection(&p2_indices).cloned().collect())
            },
            (p1, p2) => {
                let p1_activations = p1.expanded(self.p1.dim());
                let p2_activations = p2.expanded(self.p2.dim());

                let z1 = l1(p1_activations.as_slice().unwrap());
                let z2 = l1(p2_activations.as_slice().unwrap());

                Dense(p1_activations * p2_activations / z1 / z2)
            },
        }
    }
}

