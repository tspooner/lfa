use crate::basis::{Composable, Projection, Projector};
use crate::geometry::{Card, Space};

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct Scale<P> {
    projector: P,
    scale: f64,
}

impl<P> Scale<P> {
    pub fn new(projector: P, scale: f64) -> Self {
        Scale {
            projector: projector,
            scale: scale,
        }
    }
}

impl<P: Space> Space for Scale<P> {
    type Value = Projection;

    fn dim(&self) -> usize { self.projector.dim() }

    fn card(&self) -> Card { self.projector.card() }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for Scale<P> {
    fn project(&self, input: &I) -> Projection {
        Projection::Dense(self.scale * self.project_expanded(input))
    }
}

impl<P> Composable for Scale<P> {}
