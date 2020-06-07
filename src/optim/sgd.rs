use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct SGD(pub f64);

impl Optimiser<Features> for SGD {
    fn step(&mut self, weights: &mut ArrayViewMut1<f64>, features: &Features) -> Result<()> {
        Ok(features.scaled_addto(self.0, weights))
    }

    fn step_scaled(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
        scale_factor: f64,
    ) -> Result<()>
    {
        Ok(features.scaled_addto(self.0 * scale_factor, weights))
    }
}
