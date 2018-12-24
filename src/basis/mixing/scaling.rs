use crate::basis::{Projector, Projection};
use crate::geometry::{Space, Card};
use std::marker::PhantomData;

pub struct Scale<I: ?Sized, P: Projector<I>> {
    scale: f64,
    projector: P,

    phantom: PhantomData<I>,
}

impl<I: ?Sized, P: Projector<I>> Scale<I, P> {
    pub fn new(scale: f64, projector: P) -> Self {
        Scale {
            scale: scale,
            projector: projector,

            phantom: PhantomData,
        }
    }
}

impl<I: ?Sized, P: Projector<I>> Space for Scale<I, P> {
    type Value = Projection;

    fn dim(&self) -> usize {
        self.projector.dim()
    }

    fn card(&self) -> Card {
        self.projector.card()
    }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for Scale<I, P> {
    fn project(&self, input: &I) -> Projection {
        Projection::Dense(self.scale * self.project_expanded(input))
    }
}
