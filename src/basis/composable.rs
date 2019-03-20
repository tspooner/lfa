use crate::{
    basis::{
        composition::*,
        fixed::Constant,
    },
    core::{Approximator, Projection},
    geometry::Space,
    LFA,
};

/// Trait for projector composition.
///
/// This trait works very much like the Rust `Iterator`; only without any required methods. By
/// implementing this on your `Projector` you gain access to a suite of constructors that can be
/// used for rapid prototyping without explicit degradation in performance.
pub trait Composable: Sized {
    /// Return an `LFA` using this `Projector` instance and a given `Approximator`.
    fn lfa<A: Approximator<Projection>>(self, approximator: A) -> LFA<Self, A> {
        LFA::new(self, approximator)
    }

    /// Return a `Stack` of this `Projector` over another.
    fn stack<P>(self, p: P) -> Stack<Self, P> { Stack::new(self, p) }

    /// Return the `Sum` of this `Projector` and another.
    fn add<P: Space>(self, p: P) -> Sum<Self, P>
    where Self: Space {
        Sum::new(self, p)
    }

    /// Return the `Sum` of this `Projector` and the `Negate`d other.
    fn subtract<P: Space>(self, p: P) -> Sum<Self, Negate<P>>
    where Self: Space {
        Sum::new(self, Negate::new(p))
    }

    /// Return the original `Projector` with all activations `Shift`ed by some `offset`.
    fn shift(self, offset: f64) -> Shift<Self> { Shift::new(self, offset) }

    /// Return the `Product` of this `Projector` and another.
    fn multiply<P: Space>(self, p: P) -> Product<Self, P>
    where Self: Space {
        Product::new(self, p)
    }

    /// Return the `Product` of this `Projector` and the `Reciprocal` of the other.
    fn divide<P: Space>(self, p: P) -> Product<Self, Reciprocal<P>>
    where Self: Space {
        Product::new(self, Reciprocal::new(p))
    }

    /// Return the original `Projector` with all activations `Scale`d by some `factor`.
    fn scale(self, factor: f64) -> Scale<Self> { Scale::new(self, factor) }

    /// Return the original `Projector` with all activations normalised in _L₁_.
    fn normalise_l1(self) -> L1Normalise<Self> { L1Normalise::new(self) }

    /// Return the original `Projector` with all activations normalised in _L₂_.
    fn normalise_l2(self) -> L2Normalise<Self> { L2Normalise::new(self) }

    /// Return the original `Projector` with all activations normalised in _Lp_.
    fn normalise_lp(self, p: u8) -> LpNormalise<Self> { LpNormalise::new(self, p) }

    /// Return the original `Projector` with all activations normalised in _L∞_.
    fn normalise_linf(self) -> LinfNormalise<Self> { LinfNormalise::new(self) }

    /// Return the a `Stack` of this `Projector` with a single constant feature term.
    fn with_constant(self) -> Stack<Self, Constant> {
        self.stack(Constant::ones(1))
    }
}

#[cfg(test)]
mod tests {
    use super::Composable;
    use crate::basis::{composition::Stack, fixed::Constant, Projector};
    use quickcheck::quickcheck;

    #[test]
    fn test_stack_constant() {
        fn prop_equivalency(input: Vec<f64>) -> bool {
            let p1 = Stack::new(Constant::zeros(10), Constant::ones(10));
            let p2 = Constant::zeros(10).stack(Constant::ones(10));

            p1.project(&input) == p2.project(&input)
        }

        quickcheck(prop_equivalency as fn(Vec<f64>) -> bool);
    }
}
