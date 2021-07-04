use std::cmp;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use fxhash::FxHasher64;

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct Feature(u64);

pub trait CanGram {
    fn run<T: Sized + Hash + Debug, F: FnMut(Feature) -> ()>(&self, s: &[T], push_feat: &mut F);
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct SkipScheme {
    pub group_a: (usize, usize),
    pub gap: (usize, usize),
    pub group_b: (usize, usize),
}

impl CanGram for SkipScheme {
    fn run<T: Sized + Hash + Debug, F: FnMut(Feature) -> ()>(&self, s: &[T], updt: &mut F) {
        let min = self.group_a.0 + self.gap.0 + self.group_b.0;
        //println!("{:?}", &s[grp_a_1..grp_a_2]);
        if s.len() < min {
            return ();
        };
        if self.gap.1 > 0 {
            let min_gap = cmp::max(1, self.gap.0);
            for x in 0..(s.len() - min + 1) {
                for grp_a_idx in (x + self.group_a.0)..(x + self.group_a.1 + 1) {
                    if grp_a_idx > s.len() {
                        break;
                    }
                    //if x != grp_a_idx {
                    //    println!("ga: {:?}", ((x, grp_a_idx), &s[x..grp_a_idx]));
                    //};
                    let group_a = &s[x..grp_a_idx];
                    for space_idx in (grp_a_idx + min_gap)..(grp_a_idx + self.gap.1 + 1) {
                        for grp_b_idx in
                            (space_idx + self.group_b.0)..(space_idx + self.group_b.1 + 1)
                        {
                            if grp_b_idx > s.len() {
                                break;
                            }

                            let group_b = &s[space_idx..grp_b_idx];
                            let mut hasher: FxHasher64 = Default::default();

                            if group_a.len() != 0 {
                                group_a.hash(&mut hasher);
                            };
                            if group_b.len() != 0 {
                                group_b.hash(&mut hasher);
                            };
                            updt(Feature(hasher.finish()));
                        }
                    }
                }
            }
        }

        if self.gap.0 == 0 {
            let a = self.group_a.0 + self.group_b.0;
            let b = self.group_a.1 + self.group_b.1;
            for x in 0..(s.len() - a + 1) {
                for _y in a..(b + 1) {
                    let y = x + _y;

                    if y > s.len() {
                        break;
                    }
                    if x != y {
                        //println!("gram: {:?}", ((x, y), &s[x..y]));
                        let mut hasher: FxHasher64 = Default::default();
                        &s[x..y].hash(&mut hasher);
                        updt(Feature(hasher.finish()));
                    };
                }
            }
        }
    }
}

impl<Cg: CanGram> CanGram for [Cg] {
    fn run<T: Sized + Hash + Debug, F: FnMut(Feature) -> ()>(&self, s: &[T], push_feat: &mut F) {
        for cg in self {
            cg.run(s, push_feat);
        }
    }
}

pub fn n_gram(n: usize) -> SkipScheme {
    SkipScheme {
        group_a: (0, 0),
        gap: (0, 0),
        group_b: (n, n),
    }
}

pub fn skipgram(a: usize, gap: (usize, usize), b: usize) -> SkipScheme {
    SkipScheme {
        group_a: (a, a),
        gap: gap,
        group_b: (b, b),
    }
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
enum BookEnds {
    Head(u64),
    Toe(u64),
}

impl BookEnds {
    fn uniq(&self) -> u64 {
        match self {
            BookEnds::Head(i) => *i,
            BookEnds::Toe(i) => !i,
        }
    }
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct BookEndsFtzr<T> {
    head: usize,
    toe: usize,
    ftzr: T,
}

pub fn book_ends<Cg: CanGram>(head_toe: (usize, usize), cg: Cg) -> BookEndsFtzr<Cg> {
    BookEndsFtzr {
        head: head_toe.0,
        toe: head_toe.1,
        ftzr: cg,
    }
}

impl<Cg: CanGram> CanGram for BookEndsFtzr<Cg> {
    fn run<T: Sized + Hash + Debug, F: FnMut(Feature) -> ()>(&self, s: &[T], push_feat: &mut F) {
        {
            let mut pf = |n: Feature| push_feat(Feature(BookEnds::Head(n.0).uniq()));
            if s.len() >= self.head {
                //println!("head {:?}", &s[..self.head]);
                self.ftzr.run(&s[..self.head], &mut pf);
            }
        };
        {
            let mut pf = |n: Feature| push_feat(Feature(BookEnds::Toe(n.0).uniq()));
            if s.len() >= self.toe {
                //println!("toe {:?}", &s[(s.len() - self.toe)..s.len()]);
                self.ftzr.run(&s[(s.len() - self.toe)..s.len()], &mut pf);
            }
        };
    }
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct EmptyFtzr;

impl CanGram for EmptyFtzr {
    fn run<T: Sized + Hash + Debug, F: FnMut(Feature) -> ()>(&self, s: &[T], push_feat: &mut F) {}
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct MultiFtzr<A, B> {
    pub a: A,
    pub b: B,
}

impl<A: CanGram, B: CanGram> CanGram for MultiFtzr<A, B> {
    fn run<T: Sized + Hash + Debug, F: FnMut(Feature) -> ()>(&self, s: &[T], push_feat: &mut F) {
        self.a.run(&s, push_feat);
        self.b.run(&s, push_feat);
    }
}
