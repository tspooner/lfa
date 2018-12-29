use crate::basis::{Projector, Composable, Projection};
use crate::geometry::{Space, Card};

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct Shift<P> {
    projector: P,
    offset: f64,
}

impl<P> Shift<P> {
    pub fn new(projector: P, offset: f64) -> Self {
        Shift {
            projector: projector,
            offset: offset,
        }
    }
}

impl<P: Space> Space for Shift<P> {
    type Value = Projection;

    fn dim(&self) -> usize {
        self.projector.dim()
    }

    fn card(&self) -> Card {
        self.projector.card()
    }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for Shift<P> {
    fn project(&self, input: &I) -> Projection {
        Projection::Dense(self.offset + self.project_expanded(input))
    }
}

impl<P> Composable for Shift<P> {}
