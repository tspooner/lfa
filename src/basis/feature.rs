use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};
use super::IndexSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Feature {
    pub index: usize,
    pub parent_indices: IndexSet,
}

impl Feature {
    pub fn union(&self, other: &Self) -> IndexSet {
        let g = &self.parent_indices;
        let h = &other.parent_indices;

        g.union(h).cloned().collect()
    }
}

impl Hash for Feature {
    fn hash<H: Hasher>(&self, state: &mut H) { self.index.hash(state); }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateFeature {
    pub relevance: f64,
    pub parent_indices: IndexSet,
}

impl CandidateFeature {
    pub fn new<T: Into<IndexSet>>(parent_indices: T) -> Self {
        CandidateFeature {
            relevance: 0.0,
            parent_indices: parent_indices.into(),
        }
    }

    pub fn from_vec(index_vec: Vec<usize>) -> Self {
        let mut parent_indices = IndexSet::new();

        for i in index_vec {
            parent_indices.insert(i);
        }

        CandidateFeature::new(parent_indices)
    }

    pub fn into_feature(self, index: usize) -> Feature {
        Feature {
            index: index,
            parent_indices: self.parent_indices,
        }
    }
}

impl Hash for CandidateFeature {
    fn hash<H: Hasher>(&self, state: &mut H) { self.parent_indices.hash(state); }
}

impl PartialOrd for CandidateFeature {
    fn partial_cmp(&self, other: &CandidateFeature) -> Option<Ordering> {
        self.relevance.partial_cmp(&other.relevance)
    }
}
