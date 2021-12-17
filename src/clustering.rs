use std::cell::Cell;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::rc::Rc;
use std::iter::Extend;

use crate::ftzrs::{CanGram, EmptyFtzr, Feature};
use crate::fuzzypoint::FuzzyPoint;
//use crate::testing;
use crate::dualiter::*;
use crate::fuzzyindex::{FuzzyIndex, ScratchPad, SearchParams};
use crate::fuzzypoint::SimplePoint;
use crate::hasfeatures::*;
use crate::metrics::{Metric, OverlapCoefficient};
use crate::utils::{get_entry, Entry};
//use crate::d
pub(crate) type AdjacencyMatric<Point: FuzzyPoint> = Vec<Entry<Feature, Point>>;

fn log(t: usize) -> f64 {
    // if the log base is ever changed, the definition of cuttoff NEEDS to use the same base
    (t as f64).ln()
}

pub(crate) fn cluster<P: FuzzyPoint>(
    am: &mut AdjacencyMatric<P>,
    overlap_cuttoff: f64,
    epochs: usize,
) -> Vec<Entry<Feature, Feature>> {
    if (overlap_cuttoff < 0.0) || (1.0 < overlap_cuttoff) {
        panic!("The overlap coefficient must be between 0.0 and 1.0 (inclusive)")
    }
    am.sort();

    let delta = 1.0 - overlap_cuttoff;
    let step = delta / epochs as f64;
    let mut current_cuttoff = 1.0;
    let mut cache: HashMap<(Feature, Feature), f64> = Default::default();
    let mut maps: Vec<Entry<Feature, usize>> = Vec::with_capacity(am.len());
    for (idx, e) in am.iter().enumerate() {
        maps.push(Entry {
            id: e.id,
            entry: idx,
        });
    }

    let mean = {
        let mut x = 0;
        for e in am.iter() {
            x += e.entry.len();
        }
        x as f64 / am.len() as f64
    };

    fn row_of<'a, P: FuzzyPoint>(feat: &Feature, am: &'a AdjacencyMatric<P>) -> &'a P {
        match get_entry(am, &feat) {
            Some(e) => &e.1.entry,
            None => panic!("Incomplete adjacency matrix!"),
        }
    }

    fn is_unchanged(idx: usize, maps: &Vec<Entry<Feature, usize>>) -> bool {
        idx == maps[idx].entry
    }

    //fn merge_feat(a: &Feature, b: &Feature, v: &mut HashMap<Feature, Rc<Cell<Feature>>>) {
    //    let idx = get_entry(maps, a).unwrap().0;
    //

    //  }

    let mut merged_so_far = 0;
    for epoch in 0..epochs {
        println!("epoch: {:?}", (epoch, current_cuttoff));

        let cuttoff: f64 = current_cuttoff / mean.ln();
        for e in am.iter() {
            let (feat_a, feats) = (e.id, &e.entry);
            let idx_a = get_entry(&maps, &feat_a).unwrap().0;
            if is_unchanged(idx_a, &maps) {
                //let a_freq = row_of(&feat_a, &am).len();
                let point_a = row_of(&feat_a, &am);
                let freq_a = point_a.len();

                let mut best: Option<(Feature, f64)> = None;
                for feat_b in feats.get_sorted_features() {
                    let point_b = row_of(&feat_b, &am);
                    let freq_b = point_b.len();
                    if freq_b >= freq_a {
                        // overlap coefficient * inverse document frequency
                        let oc_x_idf =
                            OverlapCoefficient.sim(feats, row_of(&feat_b, &am)) / log(freq_b);
                        if oc_x_idf >= cuttoff {
                            match best {
                                None => {
                                    best = Some((*feat_b, oc_x_idf));
                                }
                                Some(p) => {
                                    if oc_x_idf > p.1 {
                                        best = Some((*feat_b, oc_x_idf));
                                    }
                                }
                            }
                        }
                    }
                }
                match best {
                    Some((feat_b, _)) => {
                        merged_so_far += 1;
                        let idx_b = get_entry(&maps, &feat_b).unwrap().0;
                        maps[idx_a].entry = idx_b;
                    }
                    None => (),
                }
            }
        }
        println!("{:?} features merged", merged_so_far);
        current_cuttoff -= step;
    }

    let mut ret: Vec<Entry<Feature, Feature>> = Vec::with_capacity(maps.len());
    for (idx_a, e) in maps.iter().enumerate() {
        if idx_a != e.entry {
            ret.push(Entry {
                id: e.id,
                entry: maps[e.entry].id,
            })
        }
    }
    ret.sort_unstable();
    ret
}

#[derive(Debug, Hash, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) struct FeatCol {
    pub(crate) id: u64,
    pub(crate) feats: Vec<Feature>,
    pub(crate) df: usize,
}

impl FuzzyPoint for FeatCol {
    fn get_sorted_features<'a>(&'a self) -> &'a [Feature] {
        &self.feats
    }
    fn get_features_mut<'a>(&'a mut self) -> &'a mut Vec<Feature> {
        &mut self.feats
    }
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct OverlapCoefficientIdf;

impl Metric<FeatCol> for OverlapCoefficientIdf {
    fn dist(&self, p1: &FeatCol, p2: &FeatCol) -> f64 {
        let sim = self.sim(p1, p2);
        if sim > 0.0 {
            1.0 / sim
        } else {
            f64::MAX
        }
    }

    #[inline(always)]
    fn sim(&self, p1: &FeatCol, p2: &FeatCol) -> f64 {
        if p1.id == p2.id {
            return 0.0;
        }
        if p1.df >= p2.df {
            return 0.0;
        }

        let mut n = 0;
        let mut left = 0;
        let mut right = 0;
        for x in DualIter::new(p1.feats.iter(), p2.feats.iter()) {
            match x {
                AndOrOr::Both(_, _) => {
                    left += 1;
                    right += 1;
                    n += 1;
                }
                AndOrOr::Left(_) => {
                    left += 1;
                }
                AndOrOr::Right(_) => {
                    right += 1;
                }
            }
        }

        let d = cmp::min(left, right);

        if n == 0 || d == 0 {
            return 0.0;
        }
        (n as f64 / d as f64) / log(cmp::max(p1.df, p2.df))
    }
}

impl HasFeatures for FeatCol {
    type Tok = ();
    fn collect_features_with<F: CanGram>(&self, ftzr: &F) -> Vec<Feature> {
        self.feats.clone()
    }
}

impl HasLabel for FeatCol {
    type Label = u64;
    fn label(&self) -> u64 {
        self.id
    }
}
#[derive(Debug, Hash, Clone, PartialOrd, Ord, PartialEq, Eq)]
struct Freq(usize);

pub(crate) fn cluster_B(
    overlap_cuttoff: f64,
    index: &FuzzyIndex<FeatCol, EmptyFtzr, FeatCol>,
    params: &SearchParams<OverlapCoefficient>,
) -> Vec<Entry<Feature, Feature>> {
    let params = params.with_metric(OverlapCoefficientIdf);
    let mut maps: Vec<Entry<Feature, usize>> = Vec::with_capacity(index.points.len());
    for p in index.points.iter() {
        maps.push(Entry {
            id: Feature(p.label),
            entry: 0,
        });
    }
    maps.sort();
    for (idx, e) in maps.iter_mut().enumerate() {
        e.entry = idx;
    }
    let mean = {
        let mut x = 0;
        for p in index.points.iter() {
            x += p.point.df;
        }
        x as f64 / index.points.len() as f64
    };

    let idf_cuttof = overlap_cuttoff / mean.ln();
    let mut changes: Vec<Entry<Freq, (Feature, Feature)>> = Vec::new();
    let mut sp = ScratchPad::new();
    for p in index.points.iter() {
        match index
            .neighbors_of_point((p.point).clone(), &mut sp, &params)
            .nearest()
        {
            None => (),
            Some(nbr) => {
                if nbr.similarity >= idf_cuttof {
                    changes.push(Entry {
                        id: Freq(p.point.df),
                        entry: (Feature(p.label), Feature(*nbr.label)),
                    });
                }
            }
        }
    }

    // This should only merge features into a feature with a greater doc frequency
    // see the sim implementation of OverlapCoefficient for FeatCol that returns 0.0 to ensure this
    // we sort by doc frequency so we will [probably] avoid cycles in the clustering graph
    changes.sort();
    changes.reverse();
    let mut same = 0;
    let mut diff = 0;
    let mut ret = Vec::with_capacity(changes.len());
    for change in changes {
        // b will always have a larger doc freq than a
        let (a, b) = change.entry;
        let idx_a = get_entry(&maps, &a).unwrap().0;
        let idx_b = get_entry(&maps, &b).unwrap().0;
        if maps[idx_b].entry == idx_b {
            maps[idx_a].entry = idx_b;
            same += 1;
        } else {
            //panic!("An error with the clustering graph has happened");
            maps[idx_a].entry = maps[idx_b].entry;
            diff += 1;
        }
    }
    println!("same/diff {:?}", (same, diff));
    for e in maps.iter() {
        let idx_b = get_entry(&maps, &e.id).unwrap().1.entry;
        let mut final_idx = idx_b;
        let mut count = index.points.len();
        while maps[final_idx].entry != final_idx {
            final_idx = maps[final_idx].entry;
            count -= 1;
            if count == 0 {
                panic!("cycle encountered in clustering");
            }
        }

        ret.push(Entry {
            id: e.id,
            entry: maps[final_idx].id,
        });
    }
    ret.sort();
    ret
}
