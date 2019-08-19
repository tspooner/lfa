use crate::{IndexT, ActivationT, Features, basis::Projector};

/// Stack the output of two `Projector` instances.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct Stacker<P1, P2> {
    p1: P1,
    p2: P2,
}

impl<P1, P2> Stacker<P1, P2> {
    pub fn new(p1: P1, p2: P2) -> Self { Stacker { p1, p2 } }
}

impl<P1, P2> Projector for Stacker<P1, P2>
where
    P1: Projector,
    P2: Projector,
{
    fn n_features(&self) -> usize {
        self.p1.n_features() + self.p2.n_features()
    }

    fn project_ith(&self, input: &[f64], index: IndexT) -> Option<ActivationT> {
        let n1 = self.p1.n_features();
        let n2 = self.p2.n_features();

        if index < n1 {
            self.p1.project_ith(input, index)
        } else if index < n1 + n2 {
            self.p2.project_ith(input, index - n1)
        } else {
            None
        }
    }

    fn project(&self, input: &[f64]) -> Features {
        self.p1.project(input).stack(self.p2.project(input))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        basis::{Projector, Constants, Indices},
        features,
    };
    use std::iter;
    use super::*;

    #[test]
    fn test_stack_constant() {
        let p = Stacker::new(Constants::new(vec![0.0; 10]), Constants::new(vec![1.0; 10]));
        let output = Features::Dense(
            iter::repeat(0.0).take(10)
                .chain(iter::repeat(1.0).take(10))
                .collect()
        );

        assert_eq!(p.n_features(), 20);

        assert_eq!(p.project(&[0.0]), output);
        assert_eq!(p.project(&[0.0, 1.0]), output);
        assert_eq!(p.project(&[-1.0, 1.0]), output);
    }

    #[test]
    fn test_stack_indices() {
        let p = Stacker::new(Indices::new(10, vec![5]), Indices::new(10, vec![0]));
        let output = Features::Sparse(20, vec![(5, 1.0), (10, 1.0)].into_iter().collect());

        assert_eq!(p.n_features(), 20);

        assert_eq!(p.project(&[0.0]), output);
        assert_eq!(p.project(&[0.0, 1.0]), output);
        assert_eq!(p.project(&[-1.0, 1.0]), output);
    }

    #[test]
    fn test_stack_mixed() {
        let p = Stacker::new(Constants::new(vec![1.0; 10]), Indices::new(10, vec![0]));
        let output = Features::Dense(
            iter::repeat(1.0).take(11)
                .chain(iter::repeat(0.0).take(9))
                .collect()
        );

        assert_eq!(p.n_features(), 20);

        assert_eq!(p.project(&[0.0]), output);
        assert_eq!(p.project(&[0.0, 1.0]), output);
        assert_eq!(p.project(&[-1.0, 1.0]), output);
    }
}
