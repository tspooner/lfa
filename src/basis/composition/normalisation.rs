use crate::basis::{Composable, Projection, Projector};
use crate::geometry::{Card, Space, norms::{l1, l2, lp, linf}};

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct L1Normalise<P>(P);

impl<P> L1Normalise<P> {
    pub fn new(projector: P) -> Self {
        L1Normalise(projector)
    }
}

impl<P: Space> Space for L1Normalise<P> {
    type Value = Projection;

    fn dim(&self) -> usize { self.0.dim() }

    fn card(&self) -> Card { self.0.card() }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for L1Normalise<P> {
    fn project(&self, input: &I) -> Projection {
        let phi = self.project_expanded(input);
        let z = l1(phi.as_slice().unwrap());

        Projection::Dense(phi / z)
    }
}

impl<P> Composable for L1Normalise<P> {}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct L2Normalise<P>(P);

impl<P> L2Normalise<P> {
    pub fn new(projector: P) -> Self {
        L2Normalise(projector)
    }
}

impl<P: Space> Space for L2Normalise<P> {
    type Value = Projection;

    fn dim(&self) -> usize { self.0.dim() }

    fn card(&self) -> Card { self.0.card() }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for L2Normalise<P> {
    fn project(&self, input: &I) -> Projection {
        let phi = self.project_expanded(input);
        let z = l2(phi.as_slice().unwrap());

        Projection::Dense(phi / z)
    }
}

impl<P> Composable for L2Normalise<P> {}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct LpNormalise<P>(P, u8);

impl<P> LpNormalise<P> {
    pub fn new(projector: P, p: u8) -> Self {
        LpNormalise(projector, p)
    }
}

impl<P: Space> Space for LpNormalise<P> {
    type Value = Projection;

    fn dim(&self) -> usize { self.0.dim() }

    fn card(&self) -> Card { self.0.card() }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for LpNormalise<P> {
    fn project(&self, input: &I) -> Projection {
        let phi = self.project_expanded(input);
        let z = lp(phi.as_slice().unwrap(), self.1);

        Projection::Dense(phi / z)
    }
}

impl<P> Composable for LpNormalise<P> {}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct LinfNormalise<P>(P);

impl<P> LinfNormalise<P> {
    pub fn new(projector: P) -> Self {
        LinfNormalise(projector)
    }
}

impl<P: Space> Space for LinfNormalise<P> {
    type Value = Projection;

    fn dim(&self) -> usize { self.0.dim() }

    fn card(&self) -> Card { self.0.card() }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for LinfNormalise<P> {
    fn project(&self, input: &I) -> Projection {
        let phi = self.project_expanded(input);
        let z = linf(phi.as_slice().unwrap());

        Projection::Dense(phi / z)
    }
}

impl<P> Composable for LinfNormalise<P> {}
