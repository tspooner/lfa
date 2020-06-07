use super::*;
use crate::{ActivationT, Error, Features, IndexT, Result};
use itertools::Itertools;
use spaces::{Card, Dim, Equipartition, ProductSpace, Space};

/// Feature prototype used by the `KernelBasis`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Prototype<C, K> {
    pub centroid: C,
    pub kernel: K,
}

impl<C: std::borrow::Borrow<[f64]>, K> Prototype<C, K> {
    pub fn distance_from<T>(&self, x: T) -> f64
    where
        T: std::borrow::Borrow<[f64]>,
        K: for<'x> kernels::Kernel<&'x [f64]>,
    {
        self.kernel.kernel(x.borrow(), self.centroid.borrow())
    }
}

/// Kernel-machine basis.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct KernelBasis<C, K> {
    pub prototypes: Vec<Prototype<C, K>>,
}

impl<C, K> KernelBasis<C, K> {
    pub fn new(prototypes: Vec<Prototype<C, K>>) -> Self { KernelBasis { prototypes } }

    pub fn homogeneous<T>(centroids: T, kernel: K) -> Self
    where
        K: Clone,
        T: IntoIterator<Item = C>,
    {
        KernelBasis::new(
            centroids
                .into_iter()
                .map(|centroid| Prototype {
                    centroid,
                    kernel: kernel.clone(),
                })
                .collect(),
        )
    }
}

impl KernelBasis<Vec<f64>, kernels::ExpQuad> {
    pub fn exp_quad(partitioning: ProductSpace<Equipartition>) -> Self {
        let lengthscales = partitioning.iter().map(|d| d.partition_width()).collect();
        let kernel = kernels::ExpQuad::new(1.0, lengthscales);
        let centroids: Vec<Vec<f64>> = partitioning
            .centres()
            .into_iter()
            .multi_cartesian_product()
            .collect();

        KernelBasis::homogeneous(centroids, kernel)
    }
}

impl KernelBasis<Vec<f64>, kernels::Matern32> {
    pub fn matern_32(partitioning: ProductSpace<Equipartition>) -> Self {
        let lengthscales = partitioning.iter().map(|d| d.partition_width()).collect();
        let kernel = kernels::Matern32::new(1.0, lengthscales);
        let centroids: Vec<Vec<f64>> = partitioning
            .centres()
            .into_iter()
            .multi_cartesian_product()
            .collect();

        KernelBasis::homogeneous(centroids, kernel)
    }
}

impl KernelBasis<Vec<f64>, kernels::Matern52> {
    pub fn matern_52(partitioning: ProductSpace<Equipartition>) -> Self {
        let lengthscales = partitioning.iter().map(|d| d.partition_width()).collect();
        let kernel = kernels::Matern52::new(1.0, lengthscales);
        let centroids: Vec<Vec<f64>> = partitioning
            .centres()
            .into_iter()
            .multi_cartesian_product()
            .collect();

        KernelBasis::homogeneous(centroids, kernel)
    }
}

impl<C, K> Space for KernelBasis<C, K> {
    type Value = Features;

    fn dim(&self) -> Dim { Dim::Finite(self.prototypes.len()) }

    fn card(&self) -> Card { Card::Infinite }
}

impl<T, K> Basis<T> for KernelBasis<Vec<f64>, K>
where
    T: std::borrow::Borrow<[f64]>,
    K: for<'x> kernels::Kernel<&'x [f64]>,
{
    fn project(&self, input: T) -> Result<Features> {
        Ok(self
            .prototypes
            .iter()
            .map(|p| p.distance_from(input.borrow()))
            .collect())
    }
}

impl<T, K> EnumerableBasis<T> for KernelBasis<Vec<f64>, K>
where
    T: std::borrow::Borrow<[f64]>,
    K: for<'x> kernels::Kernel<&'x [f64]>,
{
    fn ith(&self, input: T, index: IndexT) -> Result<ActivationT> {
        self.prototypes
            .get(index)
            .map(|p| p.distance_from(input.borrow()))
            .ok_or_else(|| Error::index_error(index, self.prototypes.len()))
    }
}

impl<C, K> Combinators for KernelBasis<C, K> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::kernels::ExpQuad;

    type Net = KernelBasis<Vec<f64>, ExpQuad>;

    fn make_net(centroids: Vec<Vec<f64>>, ls: Vec<f64>) -> Net {
        let kernel = ExpQuad::new(1.0, ls);

        KernelBasis::homogeneous(centroids, kernel)
    }

    fn make_net_1d(centroids: Vec<f64>, ls: f64) -> Net {
        make_net(centroids.into_iter().map(|c| vec![c]).collect(), vec![ls])
    }

    #[test]
    fn test_n_features() {
        fn test(net: Net, n: usize) {
            let nf: usize = net.dim().into();

            assert_eq!(nf, n);
        }

        test(make_net_1d(vec![0.0], 0.25), 1);
        test(make_net_1d(vec![0.0, 0.5, 1.0], 0.25), 3);
        test(make_net_1d(vec![0.0; 10], 0.25), 10);
        test(make_net_1d(vec![0.0; 100], 0.25), 100);
    }

    #[test]
    fn test_projection_1d() {
        let net = make_net_1d(vec![0.0, 0.5, 1.0], 0.25);
        let p = net.project([0.25]).unwrap().into_dense();

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
        let p = net.project([0.67, -7.0]).unwrap().into_dense();

        assert!((p[0] - 0.0089491).abs() < 1e-6);
        assert!((p[1] - 0.7003325).abs() < 1e-6);
        assert!((p[2] - 0.369280).abs() < 1e-6);
    }
}
