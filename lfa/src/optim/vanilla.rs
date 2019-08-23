use super::*;

pub struct SGD(pub f64);

impl Optimiser<Features> for SGD {
    fn step(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
        loss: f64
    ) -> UpdateResult<()>
    {
        Ok(features.scaled_addto(self.0 * loss, weights))
    }

    // fn step_batch(
        // &mut self,
        // weights: &mut ArrayViewMut1<f64>,
        // samples: &[(Features, f64)],
    // ) -> UpdateResult<()>
    // {
        // let sample_iter = samples.iter();
        // let (f, e) = sample_iter.next().unwrap();
        // let update = f.clone() * e;
        // let update = sample_iter.fold(update, |acc, (f, e)| {
            // acc.merge(f, |x, y| *x = *x + *y * e)
        // });

        // Ok(update.addto(&mut weights))
    // }
}