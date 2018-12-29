use crate::basis::{fixed::Constant, Composable, Projection, Projector};
use crate::geometry::{norms::l1, Card, Space};

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct Negate<P> {
    projector: P,
}

impl<P> Negate<P> {
    pub fn new(projector: P) -> Self {
        Negate {
            projector: projector,
        }
    }
}

impl<P: Space> Space for Negate<P> {
    type Value = Projection;

    fn dim(&self) -> usize { self.projector.dim() }

    fn card(&self) -> Card { self.projector.card() }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for Negate<P> {
    fn project(&self, input: &I) -> Projection {
        Projection::Dense(-self.projector.project_expanded(input))
    }
}

impl<P> Composable for Negate<P> {}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct Sum<P1, P2> {
    p1: P1,
    p2: P2,
}

impl<P1: Space, P2: Space> Sum<P1, P2> {
    pub fn new(p1: P1, p2: P2) -> Self {
        if p1.dim() != p2.dim() {
            panic!("Projectors p1 and p2 must have the same dimensionality.");
        }

        Sum { p1, p2 }
    }
}

impl<P1: Space, P2: Space> Sum<P1, Negate<P2>> {
    pub fn subtract(p1: P1, p2: P2) -> Self { Self::new(p1, Negate::new(p2)) }
}

impl<P: Space> Sum<P, Constant> {
    pub fn with_constant(projector: P, offset: f64) -> Self {
        let p2 = Constant::new(projector.dim(), offset);

        Sum {
            p1: projector,
            p2: p2,
        }
    }
}

impl<P1: Space, P2: Space> Space for Sum<P1, P2> {
    type Value = Projection;

    fn dim(&self) -> usize { self.p1.dim().max(self.p2.dim()) }

    fn card(&self) -> Card { self.p1.card() * self.p2.card() }
}

impl<I: ?Sized, P1: Projector<I>, P2: Projector<I>> Projector<I> for Sum<P1, P2> {
    fn project(&self, input: &I) -> Projection {
        let p1 = self.p1.project(input);
        let p2 = self.p2.project(input);

        match (p1, p2) {
            (Projection::Sparse(p1_indices), Projection::Sparse(p2_indices)) => {
                Projection::Sparse(p1_indices.union(&p2_indices).cloned().collect())
            },
            (p1, p2) => {
                let p1_activations = p1.expanded(self.p1.dim());
                let p2_activations = p2.expanded(self.p2.dim());

                Projection::Dense(p1_activations + p2_activations)
            },
        }
    }
}

impl<P1, P2> Composable for Sum<P1, P2> {}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct Product<P1, P2> {
    p1: P1,
    p2: P2,
}

impl<P1: Space, P2: Space> Product<P1, P2> {
    pub fn new(p1: P1, p2: P2) -> Self {
        if p1.dim() != p2.dim() {
            panic!("Projectors p1 and p2 must have the same dimensionality.");
        }

        Product { p1, p2 }
    }
}

impl<P: Space> Product<P, Constant> {
    pub fn with_factor(projector: P, factor: f64) -> Self {
        let p2 = Constant::new(projector.dim(), factor);

        Product {
            p1: projector,
            p2: p2,
        }
    }
}

impl<P1: Space, P2: Space> Space for Product<P1, P2> {
    type Value = Projection;

    fn dim(&self) -> usize { self.p1.dim() }

    fn card(&self) -> Card { unimplemented!() }
}

impl<I: ?Sized, P1: Projector<I>, P2: Projector<I>> Projector<I> for Product<P1, P2> {
    fn project(&self, input: &I) -> Projection {
        let p1 = self.p1.project(input);
        let p2 = self.p2.project(input);

        match (p1, p2) {
            (Projection::Sparse(p1_indices), Projection::Sparse(p2_indices)) => {
                Projection::Sparse(p1_indices.intersection(&p2_indices).cloned().collect())
            },
            (p1, p2) => {
                let p1_activations = p1.expanded(self.p1.dim());
                let p2_activations = p2.expanded(self.p2.dim());

                let z1 = l1(p1_activations.as_slice().unwrap());
                let z2 = l1(p2_activations.as_slice().unwrap());

                Projection::Dense(p1_activations * p2_activations / z1 / z2)
            },
        }
    }
}

impl<P1, P2> Composable for Product<P1, P2> {}
