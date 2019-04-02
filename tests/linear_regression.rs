extern crate lfa;
extern crate rand;

// use rand::{distributions::Uniform, Rng, thread_rng};
// use self::lfa::{
    // basis::{fixed::Polynomial, Composable},
    // core::{Approximator, LinearApproximator, Parameterised},
    // transforms::Exp,
    // LFA, TransformedLFA,
// };
// use std::ops::AddAssign;

// #[test]
// fn scalar() {
    // const M: f64 = 0.1;
    // const C: f64 = -0.05;

    // let mut fa = LFA::scalar(Polynomial::new(1, vec![(0.0, 1.0)]).with_constant());
    // let mut rng = thread_rng();

    // for x in rng.sample_iter(&Uniform::new_inclusive(-1.0, 1.0)).take(1000) {
        // let y_exp = M*x + C;
        // let y_apx = fa.evaluate(&vec![x]).unwrap();

        // fa.update(&vec![x], (y_exp - y_apx) * 0.1);
    // }

    // let weights = fa.weights();
    // let weights = weights.column(0);

    // println!("{:?}", weights);
    // println!("{:?}", fa.evaluate(&vec![0.0]));
    // println!("{:?}", fa.evaluate(&vec![0.5]));
    // println!("{:?}", fa.evaluate(&vec![1.0]));
    // assert!(weights.all_close(&vec![M, C].into(), 1e-3))
// }

// #[test]
// fn scalar_manual() {
    // const M: f64 = 0.1;
    // const C: f64 = -0.05;

    // let mut fa = LFA::scalar(Polynomial::new(1, vec![(0.0, 1.0)]).with_constant());
    // let mut rng = thread_rng();

    // for x in rng.sample_iter(&Uniform::new_inclusive(0.0, 1.0)).take(1000) {
        // let y_exp = M*x + C;
        // let y_apx = fa.evaluate(&vec![x]).unwrap();
        // let update = fa.compute_update_col(&vec![x], (y_exp - y_apx) * 0.1, 0);

        // fa.weights_view_mut().column_mut(0).add_assign(&update);
    // }

    // let weights = fa.weights();
    // let weights = weights.column(0);

    // assert!(weights.all_close(&vec![M, C].into(), 1e-3))
// }
