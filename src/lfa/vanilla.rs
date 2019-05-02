use crate::{
    basis::Projector,
    core::*,
    eval::*,
    geometry::{Matrix, MatrixView, MatrixViewMut, Space},
};

macro_rules! impl_builder {
    ($ftype:ty => $fname:ident) => {
        impl<P: Space> LFA<P, $ftype> {
            pub fn $fname(projector: P) -> Self {
                let evaluator = <$ftype>::zeros(projector.dim());

                Self::new(projector, evaluator)
            }
        }

        impl<P: Space> From<P> for LFA<P, $ftype> {
            fn from(projector: P) -> Self { Self::$fname(projector) }
        }
    };
}

impl_builder!(ScalarFunction => scalar);
impl_builder!(PairFunction => pair);
impl_builder!(TripleFunction => triple);

/// Linear function approximator.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct LFA<P, E> {
    pub projector: P,
    pub evaluator: E,
}

impl<P, E> LFA<P, E> {
    pub fn new(projector: P, evaluator: E) -> Self {
        LFA { projector, evaluator, }
    }
}

impl<P: Space> LFA<P, VectorFunction> {
    pub fn vector(projector: P, n_outputs: usize) -> Self {
        let evaluator = VectorFunction::zeros(projector.dim(), n_outputs);

        Self::new(projector, evaluator)
    }
}

impl<P, E: Parameterised> Parameterised for LFA<P, E> {
    fn weights(&self) -> Matrix<f64> { self.evaluator.weights() }

    fn weights_view(&self) -> MatrixView<f64> { self.evaluator.weights_view() }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> { self.evaluator.weights_view_mut() }

    fn weights_dim(&self) -> (usize, usize) { self.evaluator.weights_dim() }
}

impl<I: ?Sized, P, E> Embedding<I> for LFA<P, E>
where
    P: Projector<I>,
    E: Approximator,
{
    fn n_features(&self) -> usize {
        self.projector.dim()
    }

    fn embed(&self, input: &I) -> Features {
        self.projector.project(input)
    }
}

impl<P, E> Approximator for LFA<P, E>
where
    E: Approximator,
{
    type Output = E::Output;

    fn n_outputs(&self) -> usize {
        self.evaluator.n_outputs()
    }

    fn evaluate(&self, features: &Features) -> EvaluationResult<Self::Output> {
        self.evaluator.evaluate(features)
    }

    fn jacobian(&self, features: &Features) -> Matrix<f64> {
        self.evaluator.jacobian(features)
    }

    fn update_grad(&mut self, grad: &Matrix<f64>, update: Self::Output) -> UpdateResult<()> {
        self.evaluator.update_grad(grad, update)
    }

    fn update(&mut self, features: &Features, update: Self::Output) -> UpdateResult<()> {
        self.evaluator.update(features, update)
    }
}
