use geometry::{Card, product::LinearSpace, Space, Surjection, discrete::Partition};
use projectors::{Projection, Projector};

/// Fixed uniform basis projector.
#[derive(Clone, Serialize, Deserialize)]
pub struct UniformGrid {
    n_features: usize,
    input_space: LinearSpace<Partition>,
}

impl UniformGrid {
    pub fn new(input_space: LinearSpace<Partition>) -> Self {
        let n_features = input_space.card().into();

        UniformGrid {
            n_features: n_features,
            input_space: input_space,
        }
    }

    fn hash(&self, input: &[f64]) -> usize {
        let mut in_it = input.iter().rev();
        let mut d_it = self.input_space.iter().rev();

        let acc = d_it.next().unwrap().map(*in_it.next().unwrap());

        d_it.zip(in_it)
            .fold(acc, |acc, (d, v)| d.map(*v) + d.density() * acc)
    }
}

impl Space for UniformGrid {
    type Value = Projection;

    fn dim(&self) -> usize {
        self.n_features
    }

    fn card(&self) -> Card {
        unimplemented!()
    }
}

impl Projector<[f64]> for UniformGrid {
    fn project(&self, input: &[f64]) -> Projection {
        vec![self.hash(input)].into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_is_sparse() {
        let ds = LinearSpace::new(vec![Partition::new(0.0, 10.0, 10)]);
        let t = UniformGrid::new(ds);
        let out = t.project(&vec![0.0]);

        match out {
            Projection::Sparse(_) => assert!(true),
            Projection::Dense(_) => assert!(false),
        }
    }

    #[test]
    fn test_1d() {
        let ds = LinearSpace::new(vec![Partition::new(0.0, 10.0, 10)]);
        let t = UniformGrid::new(ds);

        assert_eq!(t.dim(), 10);

        for i in 0..10 {
            let out = t.project(&vec![i as u32 as f64]);
            let expected_bin = i;

            match out {
                Projection::Sparse(ref idx) => {
                    assert_eq!(idx.len(), 1);
                    assert!(idx.contains(&expected_bin));
                }
                _ => assert!(false),
            }

            let mut dense = arr1(&vec![0.0; 10]);
            dense[expected_bin] = 1.0;

            assert_eq!(out.expanded(t.dim()), dense);
        }
    }

    #[test]
    fn test_2d() {
        let ds = LinearSpace::new(vec![Partition::new(0.0, 10.0, 10); 2]);
        let t = UniformGrid::new(ds);

        assert_eq!(t.dim(), 100);

        for i in 0..10 {
            for j in 0..10 {
                let out = t.project(&vec![i as u32 as f64, j as u32 as f64]);
                let expected_bin = j * 10 + i;

                match out {
                    Projection::Sparse(ref idx) => {
                        assert_eq!(idx.len(), 1);
                        assert!(idx.contains(&expected_bin));
                    }
                    _ => assert!(false),
                }

                let mut dense = arr1(&vec![0.0; 100]);
                dense[expected_bin] = 1.0;

                assert_eq!(out.expanded(t.dim()), dense);
            }
        }
    }

    #[test]
    fn test_3d() {
        let ds = LinearSpace::new(vec![Partition::new(0.0, 10.0, 10); 3]);
        let t = UniformGrid::new(ds);

        assert_eq!(t.dim(), 1000);

        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let out = t.project(&vec![i as u32 as f64, j as u32 as f64, k as u32 as f64]);
                    let expected_bin = k * 100 + j * 10 + i;

                    match out {
                        Projection::Sparse(ref idx) => {
                            assert_eq!(idx.len(), 1);
                            assert!(idx.contains(&expected_bin));
                        }
                        _ => assert!(false),
                    }

                    let mut dense = arr1(&vec![0.0; 1000]);
                    dense[expected_bin] = 1.0;

                    assert_eq!(out.expanded(t.dim()), dense);
                }
            }
        }
    }
}
