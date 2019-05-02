extern crate lfa;
extern crate rand;

use rand::{distributions::Uniform, Rng, thread_rng};
use self::lfa::{
    basis::fixed::Polynomial,
    composition::Composable,
    core::{Parameterised, Approximator, Embedding},
    LFA,
};

#[test]
fn scalar() {
    const M: f64 = 0.1;
    const C: f64 = -0.05;

    let mut fa = LFA::scalar(Polynomial::new(1, vec![(-1.0, 1.0)]).with_constant());
    let mut rng = thread_rng();

    for x in rng.sample_iter(&Uniform::new_inclusive(-1.0, 1.0)).take(1000) {
        let y_exp = M*x + C;

        let x = fa.embed(&vec![x]);
        let y_apx = fa.evaluate(&x).unwrap();

        fa.update(&x, (y_exp - y_apx) * 0.1).ok();
    }

    let weights = fa.weights();
    let weights = weights.column(0);

    assert!(weights.all_close(&vec![M, C].into(), 1e-3))
}

#[test]
fn scalar_manual() {
    const M: f64 = 0.1;
    const C: f64 = -0.05;

    let mut fa = LFA::scalar(Polynomial::new(1, vec![(-1.0, 1.0)]).with_constant());
    let mut rng = thread_rng();

    for x in rng.sample_iter(&Uniform::new_inclusive(-1.0, 1.0)).take(1000) {
        let y_exp = M*x + C;

        let x = fa.embed(&vec![x]);
        let y_apx = fa.evaluate(&x).unwrap();

        fa.weights_view_mut().column_mut(0).scaled_add((y_exp - y_apx) * 0.1, &x.expanded(2));
    }

    let weights = fa.weights();
    let weights = weights.column(0);

    assert!(weights.all_close(&vec![M, C].into(), 1e-3))
}
