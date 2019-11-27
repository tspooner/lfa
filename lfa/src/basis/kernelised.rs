use crate::{
    IndexT, ActivationT, Features, Result, Error,
    basis::{Basis, kernels::{self, Kernel}},
};
use itertools::Itertools;
use spaces::{Equipartition, ProductSpace};

/// Feature prototype used by the `KernelBasis`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Prototype<K: Kernel<[f64]>> {
    pub centroid: Vec<f64>,
    pub kernel: K,
}

impl<K: Kernel<[f64]>> Prototype<K> {
    pub fn kernel(&self, x: &[f64]) -> f64 { self.kernel.kernel(x, &self.centroid) }
}

/// Kernel-machine basis.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct KernelBasis<K: Kernel<[f64]>> {
    pub prototypes: Vec<Prototype<K>>,
}

impl<K: Kernel<[f64]>> KernelBasis<K> {
    pub fn new(prototypes: Vec<Prototype<K>>) -> Self { KernelBasis { prototypes } }

    pub fn homogeneous<T, I>(centroids: T, kernel: K) -> Self
    where
        T: IntoIterator<Item = I>,
        I: Into<Vec<f64>>,
    {
        KernelBasis::new(
            centroids
                .into_iter()
                .map(|c| Prototype {
                    centroid: c.into(),
                    kernel: kernel.clone(),
                })
                .collect(),
        )
    }
}

impl KernelBasis<kernels::ExpQuad> {
    pub fn exp_quad(partitioning: ProductSpace<Equipartition>) -> Self {
        let lengthscales = partitioning.iter().map(|d| d.partition_width()).collect();
        let kernel = kernels::ExpQuad::new(1.0, lengthscales);
        let centroids: Vec<Vec<_>> =
            partitioning.centres().into_iter().multi_cartesian_product().collect();

        KernelBasis::homogeneous(centroids, kernel)
    }
}

impl KernelBasis<kernels::Matern32> {
    pub fn matern_32(partitioning: ProductSpace<Equipartition>) -> Self {
        let lengthscales = partitioning.iter().map(|d| d.partition_width()).collect();
        let kernel = kernels::Matern32::new(1.0, lengthscales);
        let centroids: Vec<Vec<_>> =
            partitioning.centres().into_iter().multi_cartesian_product().collect();

        KernelBasis::homogeneous(centroids, kernel)
    }
}

impl KernelBasis<kernels::Matern52> {
    pub fn matern_52(partitioning: ProductSpace<Equipartition>) -> Self {
        let lengthscales = partitioning.iter().map(|d| d.partition_width()).collect();
        let kernel = kernels::Matern52::new(1.0, lengthscales);
        let centroids: Vec<Vec<_>> =
            partitioning.centres().into_iter().multi_cartesian_product().collect();

        KernelBasis::homogeneous(centroids, kernel)
    }
}

impl<K: Kernel<[f64]>> Basis for KernelBasis<K> {
    fn n_features(&self) -> usize {
        self.prototypes.len()
    }

    fn project_ith(&self, input: &[f64], index: IndexT) -> Result<Option<ActivationT>> {
        self.prototypes.get(index)
            .map(|p| Some(p.kernel(input)))
            .ok_or_else(|| Error::index_error(index, self.prototypes.len()))
    }

    fn project(&self, input: &[f64]) -> Result<Features> {
        Ok(self.prototypes.iter().map(|p| p.kernel(input)).collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::basis::kernels::ExpQuad;
    use super::*;

    fn make_net(centroids: Vec<Vec<f64>>, ls: Vec<f64>) -> KernelBasis<ExpQuad> {
        let kernel = ExpQuad::new(1.0, ls);

        KernelBasis::homogeneous(centroids, kernel)
    }

    fn make_net_1d(centroids: Vec<f64>, ls: f64) -> KernelBasis<ExpQuad> {
        make_net(centroids.into_iter().map(|c| vec![c]).collect(), vec![ls])
    }

    #[test]
    fn test_n_features() {
        assert_eq!(make_net_1d(vec![0.0], 0.25).n_features(), 1);
        assert_eq!(make_net_1d(vec![0.0, 0.5, 1.0], 0.25).n_features(), 3);
        assert_eq!(make_net_1d(vec![0.0; 10], 0.25).n_features(), 10);
        assert_eq!(make_net_1d(vec![0.0; 100], 0.25).n_features(), 100);
    }

    #[test]
    fn test_projection_1d() {
        let net = make_net_1d(vec![0.0, 0.5, 1.0], 0.25);
        let p = net.project(&[0.25]).unwrap().expanded();

        assert!((p[0] - 0.6065307).abs() < 1e-6);
        assert!((p[1] - 0.6065307).abs() < 1e-6);
        assert!((p[2] - 0.0111090).abs() < 1e-6);
    }

    #[test]
    fn test_projection_2d() {
        let net = make_net(
            vec![vec![0.0, -10.0], vec![0.5, -8.0], vec![1.0, -6.0]],
            vec![0.25, 2.0],
        );
        let p = net.project(&[0.67, -7.0]).unwrap().expanded();

        assert!((p[0] - 0.0089491).abs() < 1e-6);
        assert!((p[1] - 0.7003325).abs() < 1e-6);
        assert!((p[2] - 0.369280).abs() < 1e-6);
    }
}
