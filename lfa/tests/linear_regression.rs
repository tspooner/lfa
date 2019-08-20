extern crate lfa;
extern crate rand;
extern crate rand_distr;

use rand::{Rng, thread_rng};
use rand_distr::Uniform;
use self::lfa::{
    Approximator, Parameterised, ScalarFunction, Features,
    basis::{Projector, Polynomial},
    sgd,
};

#[test]
fn scalar_sgd() {
    const M: f64 = 0.1;
    const C: f64 = -0.05;

    let basis = Polynomial::new(1, 1).with_constant();

    let mut fa = ScalarFunction::zeros(basis.n_features());
    let mut opt = sgd::SGD(1.0);

    for x in thread_rng().sample_iter(&Uniform::new_inclusive(-1.0, 1.0)).take(1000) {
        let y_exp = M*x + C;

        let x = basis.project(&vec![x]);
        let y_apx = fa.evaluate(&x).unwrap();

        fa.update_with(&mut opt, &x, y_exp - y_apx).ok();
    }

    let weights = fa.weights();
    let weights = weights.column(0);

    assert!(weights.all_close(&vec![M, C].into(), 1e-5))
}
