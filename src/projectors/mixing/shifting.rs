use geometry::{Space, Card};
use projectors::{Projector, Projection};
use std::marker::PhantomData;

pub struct Shift<I: ?Sized, P: Projector<I>> {
    offset: f64,
    projector: P,

    phantom: PhantomData<I>,
}

impl<I: ?Sized, P: Projector<I>> Shift<I, P> {
    pub fn new(offset: f64, projector: P) -> Self {
        Shift {
            offset: offset,
            projector: projector,

            phantom: PhantomData,
        }
    }
}

impl<I: ?Sized, P: Projector<I>> Space for Shift<I, P> {
    type Value = Projection;

    fn dim(&self) -> usize {
        self.projector.dim()
    }

    fn card(&self) -> Card {
        self.projector.card()
    }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for Shift<I, P> {
    fn project(&self, input: &I) -> Projection {
        Projection::Dense(self.offset + self.project_expanded(input))
    }
}
