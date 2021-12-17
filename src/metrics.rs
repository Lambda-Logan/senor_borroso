use std::cell::Cell;
use std::cmp;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use crate::dualiter::*;
use crate::fuzzypoint::{FuzzyPoint, SimHash};

pub trait Metric<P> {
    fn sim(&self, p1: &P, p2: &P) -> f64;
    fn dist(&self, p1: &P, p2: &P) -> f64;
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct Jaccard;

impl<P: FuzzyPoint> Metric<P> for Jaccard {
    fn dist(&self, p1: &P, p2: &P) -> f64 {
        let sim = self.sim(p1, p2);
        if sim > 0.0 {
            1.0 / sim
        } else {
            f64::MAX
        }
    }

    #[inline(always)]
    fn sim(&self, p1: &P, p2: &P) -> f64 {
        //return 1.0 / self.ham_dist(other) as f64;
        let mut n = 0;
        let mut d = 0;
        for x in DualIter::new(
            p1.get_sorted_features().iter(),
            p2.get_sorted_features().iter(),
        ) {
            match x {
                AndOrOr::Both(_, _) => {
                    d += 1;
                    n += 1;
                }
                _ => {
                    d += 1;
                }
            }
        }
        if n == 0 || d == 0 {
            return 0.0;
        }
        n as f64 / d as f64
    }
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct OverlapCoefficient;

impl<Point: FuzzyPoint> Metric<Point> for OverlapCoefficient {
    fn dist(&self, p1: &Point, p2: &Point) -> f64 {
        let sim = self.sim(p1, p2);
        if sim > 0.0 {
            1.0 / sim
        } else {
            f64::MAX
        }
    }

    #[inline(always)]
    fn sim(&self, p1: &Point, p2: &Point) -> f64 {
        let mut n = 0;
        let mut left = 0;
        let mut right = 0;
        for x in DualIter::new(
            p1.get_sorted_features().iter(),
            p2.get_sorted_features().iter(),
        ) {
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
        n as f64 / d as f64
    }
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct Counted<M> {
    pub metric: M,
    pub tally: Rc<Cell<usize>>,
}

impl<M> Counted<M> {
    pub fn new(m: M) -> Self {
        Counted {
            metric: m,
            tally: Rc::new(Cell::new(0)),
        }
    }
}

impl<P, M: Metric<P>> Metric<P> for Counted<M> {
    fn sim(&self, p1: &P, p2: &P) -> f64 {
        let t = self.tally.get();
        self.tally.set(t + 1);
        self.metric.sim(p1, p2)
    }
    fn dist(&self, p1: &P, p2: &P) -> f64 {
        let t = self.tally.get();
        self.tally.set(t + 1);
        self.metric.dist(p1, p2)
    }
}

#[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct Hamming;

impl Metric<SimHash> for Hamming {
    fn sim(&self, p1: &SimHash, p2: &SimHash) -> f64 {
        (1.0 / (self.dist(p1, p2) + 1.0)).sqrt()
    }
    fn dist(&self, p1: &SimHash, p2: &SimHash) -> f64 {
        (p1.hash ^ p2.hash).count_ones() as f64
    }
}
