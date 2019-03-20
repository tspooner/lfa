use crate::{
    basis::{
        fixed::Constant,
        Composable,
    },
    core::{Projection, Projector},
    geometry::{Card, Space},
};

/// Apply negation to the output of a `Projector` instance.
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct Negate<P>(P);

impl<P> Negate<P> {
    pub fn new(projector: P) -> Self {
        Negate(projector)
    }
}

impl<P: Space> Space for Negate<P> {
    type Value = Projection;

    fn dim(&self) -> usize { self.0.dim() }

    fn card(&self) -> Card { self.0.card() }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for Negate<P> {
    fn project(&self, input: &I) -> Projection {
        Projection::Dense(-self.0.project_expanded(input))
    }
}

impl<P> Composable for Negate<P> {}

/// Sum the output of two `Projector` instances.
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

/// Apply inversion to the output of a `Projector` instance.
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct Reciprocal<P>(P);

impl<P: Space> Reciprocal<P> {
    pub fn new(projector: P) -> Self {
        Reciprocal(projector)
    }
}

impl<P: Space> Space for Reciprocal<P> {
    type Value = Projection;

    fn dim(&self) -> usize { self.0.dim() }

    fn card(&self) -> Card { unimplemented!() }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for Reciprocal<P> {
    fn project(&self, input: &I) -> Projection {
        let p = self.0.project(input);

        match p {
            Projection::Sparse(_) => p,
            Projection::Dense(activations) => Projection::Dense(1.0 / activations),
        }
    }
}

impl<P> Composable for Reciprocal<P> {}

/// Multiply the output of two `Projector` instances.
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

                Projection::Dense(p1_activations * p2_activations)
            },
        }
    }
}

impl<P1, P2> Composable for Product<P1, P2> {}

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;
    use super::*;

    #[test]
    fn test_negate() {
        fn prop_output(length: usize, value: f64) -> bool {
            let p = Negate::new(Constant::new(length, value));

            p.project_expanded(&[0.0]).into_iter().all(|&v| v == -value)
        }

        quickcheck(prop_output as fn(usize, f64) -> bool);
    }

    #[test]
    fn test_addition() {
        fn prop_output(length: usize, v1: f64, v2: f64) -> bool {
            let p = Sum::new(Constant::new(length, v1), Constant::new(length, v2));

            p.project_expanded(&[0.0]).into_iter().all(|&v| v == v1 + v2)
        }

        quickcheck(prop_output as fn(usize, f64, f64) -> bool);
    }

    #[test]
    fn test_subtraction() {
        fn prop_output(length: usize, v1: f64, v2: f64) -> bool {
            let p = Sum::new(Constant::new(length, v1), Negate::new(Constant::new(length, v2)));

            p.project_expanded(&[0.0]).into_iter().all(|&v| v == v1 - v2)
        }

        quickcheck(prop_output as fn(usize, f64, f64) -> bool);
    }

    #[test]
    fn test_reciprocal() {
        fn prop_output(length: usize, value: f64) -> bool {
            let p = Reciprocal::new(Constant::new(length, value));

            p.project_expanded(&[0.0]).into_iter().all(|&v| v == 1.0 / value)
        }

        quickcheck(prop_output as fn(usize, f64) -> bool);
    }

    #[test]
    fn test_multiplication() {
        fn prop_output(length: usize, v1: f64, v2: f64) -> bool {
            let p = Product::new(Constant::new(length, v1), Constant::new(length, v2));

            p.project_expanded(&[0.0]).into_iter().all(|&v| (v - v1 * v2) < 1e-7)
        }

        quickcheck(prop_output as fn(usize, f64, f64) -> bool);
    }

    #[test]
    fn test_division() {
        fn prop_output(length: usize, v1: f64, v2: f64) -> bool {
            let p = Product::new(
                Constant::new(length, v1),
                Reciprocal::new(Constant::new(length, v2))
            );

            p.project_expanded(&[0.0]).into_iter().all(|&v| (v - v1 / v2) < 1e-7)
        }

        quickcheck(prop_output as fn(usize, f64, f64) -> bool);
    }
}
