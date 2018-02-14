#[derive(Clone, Copy)]
pub struct Pipeline {}


pub trait Transformer<I, O> {
    fn transform(&self, input: I) -> Option<O>;
}

mod bounds;
pub use self::bounds::*;
