use super::{Projector, Projection};


/// Fixed uniform basis projector.
#[derive(Clone, Serialize, Deserialize)]
pub struct UniformGrid {
    bounds: Vec<(f64, f64)>,
    densities: Vec<usize>,
}

impl UniformGrid {
    pub fn new(bounds: Vec<(f64, f64)>, densities: Vec<usize>) -> Self {
        UniformGrid {
            bounds: bounds,
            densities: densities,
        }
    }

    fn hash(&self, input: &[f64]) -> usize {
        let mut in_it = input.iter().rev();
        let mut d_it = self.input_space.iter().rev();

        let acc = d_it.next().unwrap().convert(*in_it.next().unwrap());

        d_it.zip(in_it)
            .fold(acc, |acc, (d, v)| d.convert(*v) + d.density() * acc)
    }
}

impl Projector<[f64]> for UniformGrid {
    fn project(&self, input: &[f64]) -> Projection {
        Projection::Sparse(vec![self.hash(input)])
    }

    fn dim(&self) -> usize {
        self.bounds.len()
    }

    fn size(&self) -> usize {
        self.n_features
    }

    fn activity(&self) -> usize {
        1
    }

    fn equivalent(&self, other: &Self) -> bool {
        self.dim() == other.dim() && self.size() == other.size()
    }
}


#[cfg(test)]
mod tests {
    use super::{Projector, Projection, UniformGrid};
    use ndarray::arr1;
    use geometry::RegularSpace;
    use geometry::dimensions::Partitioned;


    #[test]
    fn test_is_sparse() {
        let ds = RegularSpace::new(vec![Partitioned::new(0.0, 10.0, 10)]);
        let t = UniformGrid::new(ds);
        let out = t.project(&vec![0.0]);

        match out {
            Projection::Sparse(_) => assert!(true),
            Projection::Dense(_) => assert!(false)
        }
    }

    #[test]
    fn test_1d() {
        let ds = RegularSpace::new(vec![Partitioned::new(0.0, 10.0, 10)]);
        let t = UniformGrid::new(ds);

        assert_eq!(t.size(), 10);

        for i in 0..10 {
            let out = t.project(&vec![i as u32 as f64]);
            let expected_bin = i;

            match out {
                Projection::Sparse(ref idx) => {
                    assert_eq!(idx.len(), 1);
                    assert_eq!(idx[0], expected_bin);
                },
                _ => assert!(false)
            }

            let mut dense = arr1(&vec![0.0; 10]);
            dense[expected_bin] = 1.0;

            assert_eq!(t.expand_projection(out), dense);
        }
    }

    #[test]
    fn test_2d() {
        let ds = RegularSpace::new(vec![Partitioned::new(0.0, 10.0, 10); 2]);
        let t = UniformGrid::new(ds);

        assert_eq!(t.size(), 100);

        for i in 0..10 {
            for j in 0..10 {
                let out = t.project(&vec![i as u32 as f64, j as u32 as f64]);
                let expected_bin = j*10 + i;

                match out {
                    Projection::Sparse(ref idx) => {
                        assert_eq!(idx.len(), 1);
                        assert_eq!(idx[0], expected_bin);
                    },
                    _ => assert!(false)
                }

                let mut dense = arr1(&vec![0.0; 100]);
                dense[expected_bin] = 1.0;

                assert_eq!(t.expand_projection(out), dense);
            }
        }
    }

    #[test]
    fn test_3d() {
        let ds = RegularSpace::new(vec![Partitioned::new(0.0, 10.0, 10); 3]);
        let t = UniformGrid::new(ds);

        assert_eq!(t.size(), 1000);

        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let out = t.project(&vec![i as u32 as f64, j as u32 as f64, k as u32 as f64]);
                    let expected_bin = k*100 + j*10 + i;

                    match out {
                        Projection::Sparse(ref idx) => {
                            assert_eq!(idx.len(), 1);
                            assert_eq!(idx[0], expected_bin);
                        },
                        _ => assert!(false)
                    }

                    let mut dense = arr1(&vec![0.0; 1000]);
                    dense[expected_bin] = 1.0;

                    assert_eq!(t.expand_projection(out), dense);
                }
            }
        }
    }
}
