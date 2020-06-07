use super::*;
use crate::{ActivationT, Error, Features, IndexT, Result};
use spaces::{Card, Dim, Space};

/// Stack the output of two `Basis` instances.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Stack<B1, B2> {
    b1: B1,
    b2: B2,
}

impl<B1, B2> Stack<B1, B2> {
    pub fn new(b1: B1, b2: B2) -> Self { Stack { b1, b2 } }
}

impl<B1: Space, B2: Space> Space for Stack<B1, B2> {
    type Value = Features;

    fn dim(&self) -> Dim { self.b1.dim() + self.b2.dim() }

    fn card(&self) -> Card { self.b1.card() * self.b2.card() }
}

impl<T, B1, B2> Basis<T> for Stack<B1, B2>
where
    T: Clone,

    B1: Basis<T, Value = Features>,
    B2: Basis<T, Value = Features>,
{
    fn project(&self, input: T) -> Result<Features> {
        self.b1
            .project(input.clone())
            .and_then(|f1| self.b2.project(input).map(|f2| f1.stack(f2)))
    }
}

impl<T, B1, B2> EnumerableBasis<T> for Stack<B1, B2>
where
    T: Clone,

    B1: EnumerableBasis<T, Value = Features>,
    B2: EnumerableBasis<T, Value = Features>,
{
    fn ith(&self, input: T, index: IndexT) -> Result<ActivationT> {
        let n1: usize = self.b1.dim().into();
        let n2: usize = self.b2.dim().into();

        let n12 = n1 + n2;

        if index < n1 {
            self.b1.ith(input, index)
        } else if index < n12 {
            self.b2.ith(input, index - n1)
        } else {
            Err(Error::index_error(index, n12))
        }
    }
}

impl<B1: Combinators, B2: Combinators> Combinators for Stack<B1, B2> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{Basis, Fixed};
    use std::iter;

    #[test]
    fn test_stack_constant() {
        let basis = Stack::new(Fixed::dense(vec![0.0; 10]), Fixed::dense(vec![1.0; 10]));
        let output = Features::Dense(
            iter::repeat(0.0)
                .take(10)
                .chain(iter::repeat(1.0).take(10))
                .collect(),
        );

        let nf: usize = basis.dim().into();

        assert_eq!(nf, 20);

        assert_eq!(basis.project(&[0.0]).unwrap(), output);
        assert_eq!(basis.project(&[0.0, 1.0]).unwrap(), output);
        assert_eq!(basis.project(&[-1.0, 1.0]).unwrap(), output);
    }

    #[test]
    fn test_stack_indices() {
        let basis = Stack::new(Fixed::sparse(10, vec![5]), Fixed::sparse(10, vec![0]));
        let output = Features::unitary(20, vec![5, 10]);

        let nf: usize = basis.dim().into();

        assert_eq!(nf, 20);

        assert_eq!(basis.project(&[0.0]).unwrap(), output);
        assert_eq!(basis.project(&[0.0, 1.0]).unwrap(), output);
        assert_eq!(basis.project(&[-1.0, 1.0]).unwrap(), output);
    }

    #[test]
    fn test_stack_mixed() {
        let basis = Stack::new(Fixed::dense(vec![1.0; 10]), Fixed::sparse(10, vec![0]));
        let output = Features::Dense(
            iter::repeat(1.0)
                .take(11)
                .chain(iter::repeat(0.0).take(9))
                .collect(),
        );

        let nf: usize = basis.dim().into();

        assert_eq!(nf, 20);

        assert_eq!(basis.project(&[0.0]).unwrap(), output);
        assert_eq!(basis.project(&[0.0, 1.0]).unwrap(), output);
        assert_eq!(basis.project(&[-1.0, 1.0]).unwrap(), output);
    }
}
