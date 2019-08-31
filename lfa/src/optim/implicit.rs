use super::*;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ISGD(pub f64);

impl Optimiser<Features> for ISGD {
    fn step(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
        loss: f64
    ) -> Result<()> {
        let norm = features.fold(0.0, |acc, x| acc + x*x);
        let lr = self.0 / (1.0 + self.0 * norm);

        Ok(features.scaled_addto(lr * loss, weights))
    }
}
