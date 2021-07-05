use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::marker::PhantomData;

use crate::ftzrs::*;
use crate::fuzzypoint::*;
use crate::hasfeatures::*;
use crate::utils::shuffle;
use crate::utils::{get_entry, Entry};

#[derive(Debug, Clone, Default)]
struct FreqCounterWith<T> {
    to_sort: Vec<(T, u32)>,
    freqs: Vec<(T, u32)>,
}

impl<T: Ord + Debug> FreqCounterWith<T> {
    pub fn new() -> FreqCounterWith<T> {
        FreqCounterWith {
            to_sort: Vec::new(),
            freqs: Vec::new(),
        }
    }

    fn frequencies<'a, Items>(&'a mut self, items: Items) -> &'a mut Vec<(T, u32)>
    where
        Items: Iterator<Item = (T, u32)>,
    {
        self.to_sort.clear();
        self.freqs.clear();

        self.to_sort.extend(items);
        self.to_sort.sort_by(|a, b| a.0.cmp(&b.0));
        let mut item_iter = self.to_sort.drain(..);

        for item in item_iter {
            match self.freqs.last_mut() {
                None => {
                    self.freqs.push(item);
                }
                Some(last) => {
                    if last.0 == item.0 {
                        last.1 += item.1;
                    } else {
                        self.freqs.push(item);
                    }
                }
            }
        }
        &mut self.freqs
    }
}

/////////////////////
/// Preallocates and holds mutable state to shave time off each lookup in the index
/// # Examples
/// ```
/// let mut sp = ScratchPad::new();
/// println(":?}", fuzzy_index.best_match("word", &mut sp));
/// ```
/// ////////////////
pub struct ScratchPad {
    fc: FreqCounterWith<u32>,
    freq_bow: Vec<Entry<usize, (usize, Feature)>>,
}
impl ScratchPad {
    pub fn new() -> Self {
        ScratchPad {
            fc: FreqCounterWith::new(),
            freq_bow: Vec::new(),
        }
    }
}

impl Default for ScratchPad {
    fn default() -> Self {
        ScratchPad::new()
    }
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct SearchParams<M> {
    pub metric: M,
    pub depth: usize,
    pub breadth: usize,
    pub timeout_n: u32,
    pub return_if_gt: u8,
    pub max_comparisons: usize,
}

impl<M1> SearchParams<M1> {
    /////////////////////
    /// returns search parameters identical to self, only with a new metric.
    /// (This is because the metric cannot be mutated in a typesafe way)
    /// # Examples
    /// ```
    /// let params1 = SearchParams {
    ///                  metric: Jaccard,
    ///                  depth: 20,
    ///                  breadth: 20,
    ///                  max_comparisons: 70,
    ///                  return_if_gt: 70,
    ///                  timeout_n: 20,
    /// };
    /// let params2 = params.with_metric(Hamming);
    /// ```
    /// //////////////////
    pub fn with_metric<M2>(&self, m: M2) -> SearchParams<M2> {
        SearchParams {
            metric: m,
            depth: self.depth,
            breadth: self.breadth,
            max_comparisons: self.max_comparisons,
            return_if_gt: self.return_if_gt,
            timeout_n: self.timeout_n,
        }
    }
}

pub struct Neighbor<Label> {
    pub label: Label,
    pub similarity: f64,
}

#[derive(Clone, Debug)]
pub struct FuzzySearchIndex<Origin, Tok, Id, Ftzr, Point> {
    pub ftzr: Ftzr,
    pub points: Vec<Labeled<Point, Id>>,
    pub all_feats: Vec<Entry<Feature, Vec<u32>>>,
    _marker: PhantomData<(Origin, Tok)>,
}

pub type FuzzyIndex<Origin: HasName + HasFeatures, Ftzr, Point: FuzzyPoint> =
    FuzzySearchIndex<Origin, <Origin as HasFeatures>::Tok, <Origin as HasName>::Id, Ftzr, Point>;

impl<Origin, Tok, Id, Ftzr, Point> FuzzySearchIndex<Origin, Tok, Id, Ftzr, Point>
where
    Tok: Sized + Hash + Debug,
    Origin: HasFeatures<Tok = Tok> + HasName<Id = Id> + Eq + Hash,
    Point: FuzzyPoint,
    Ftzr: CanGram,
    Id: Clone,
{
    pub fn new<W: Iterator<Item = Origin>>(ftzr: Ftzr, sequences: W) -> Self {
        let words_set: HashSet<Origin> = Iterator::collect(sequences);
        // sort points set???
        let mut points: Vec<Labeled<Point, Id>> =
            Iterator::collect(words_set.into_iter().map(|x| Labeled {
                label: x.name(),
                point: Point::made_from(&x, &ftzr),
            }));

        let mut feats_hm: HashMap<Feature, Vec<u32>> = HashMap::with_capacity(points.len());
        let empty = Vec::new();
        for (idx, labeled_point) in points.iter().enumerate() {
            for feat in labeled_point.point.get_sorted_features().iter() {
                let v = feats_hm.entry(feat.clone()).or_insert(empty.clone());
                v.push(idx as u32);
            }
        }
        let mut feats: Vec<Entry<Feature, Vec<u32>>> = Vec::with_capacity(feats_hm.len());

        for (feat, idxs) in feats_hm.into_iter() {
            feats.push(Entry {
                id: feat,
                entry: shuffle(&idxs),
            });
        }
        feats.sort();
        //TODO randomize order of the Vec<u32>'s in toks??
        //println!("w t {:?}", (points.len(), toks.len()));
        let mut r = FuzzySearchIndex {
            ftzr: ftzr,
            points: points,
            all_feats: feats,
            _marker: Default::default(),
        };
        r
    }

    fn freq_of(&self, f: &Feature) -> usize {
        get_entry(&self.all_feats, f)
            .map(|(_, e)| e.entry.len())
            .unwrap_or(0)
    }

    ///////////////////////
    /// Only indexes the 'n' rarest features of each point in the index.
    /// The rest are discarded from the index.
    /// This can result in faster lookups and higher accuracy.
    /// (Only the index of features is changed, not the individual points)
    /// Do not use when erroneous features are present the points given to create FuzzyIndex.
    ///////////////////////
    pub fn compress_index(&mut self, cuttoff: usize) {
        let mut v = vec![cuttoff];
        for _ in 0..4 {
            let phi = (v.last().unwrap() * 144) / 89;
            v.push(phi);
        }
        v.reverse();
        for ctf in v {
            self.compress_step(ctf);
        }
    }

    fn compress_step(&mut self, cuttoff: usize) {
        let mut too_common: Vec<Entry<(usize, Feature), ()>> =
            Vec::with_capacity(self.all_feats.len());
        let mut toks_sort: Vec<Feature> = Vec::new();
        for (idx, labeled_point) in self.points.iter().enumerate() {
            if labeled_point.point.get_sorted_features().len() > cuttoff {
                toks_sort.extend(
                    labeled_point
                        .point
                        .get_sorted_features()
                        .iter()
                        .map(|f| f.clone()),
                );
                toks_sort.sort_by_cached_key(|t| self.freq_of(t));
                for tok in toks_sort[cuttoff..].iter() {
                    too_common.push(Entry {
                        id: (idx, tok.clone()),
                        entry: (),
                    });
                }
                toks_sort.clear();
            }
        }

        too_common.sort();

        let mut sort_slate: &mut Vec<u32> = &mut Vec::new();
        for e in self.all_feats.iter_mut() {
            let mut points = &mut e.entry;
            sort_slate.extend(points.iter());
            points.clear();
            for word_idx in sort_slate.iter() {
                if get_entry(&too_common, &(*word_idx as usize, e.id.clone())).is_none() {
                    points.push(*word_idx);
                }
            }
            sort_slate.clear();
        }
    }

    pub fn best_match<M: Metric<Point>>(
        &self,
        tokens: Origin,
        sp: &mut ScratchPad,
        params: &SearchParams<M>,
    ) -> Option<(Id, f64)> {
        let point = Point::made_from(&tokens, &self.ftzr);
        sp.freq_bow.clear(); //Vec::with_capacity(point.get_sorted_features().len());
                             // 36 ms
        sp.freq_bow
            .extend(point.get_sorted_features().iter().filter_map(|a| {
                let r = get_entry(&self.all_feats, &a).map(|(idx, e)| Entry {
                    id: e.entry.len(),
                    entry: (idx, *a),
                });
                r
            }));
        sp.freq_bow.sort();
        // 25 μs
        let n_toks = point.get_sorted_features().len() as f64;
        let mut r: Option<(Id, f64)> = None;
        let mut max_so_far: f64 = 0.0;

        let word_idxs = sp
            .freq_bow
            .iter()
            .take(params.depth) //48
            .map(|e| {
                let tfidf: u32 = 512 / ((e.id as f64).log2() as u32 + 1);
                (self.all_feats[e.entry.0 as usize])
                    .entry
                    .iter()
                    .take(params.breadth)
                    .map(move |entry| (*entry, 1))
            })
            .flatten();

        let words_freqs = sp.fc.frequencies(word_idxs);
        // 47 μs
        words_freqs.sort_unstable_by(|a, b| (b.1).cmp(&a.1));
        // 51 μs
        let mut last_win_ago = 0;
        let return_if_gt = params.return_if_gt as f64 / 100.0;
        for (word_idx, _) in words_freqs.iter().take(params.max_comparisons) {
            if last_win_ago > params.timeout_n {
                break;
            }
            if true {
                let word_b_info = &self.points[*word_idx as usize];
                let jaccard = params.metric.sim(&point, &word_b_info.point);
                if jaccard > return_if_gt {
                    return Some((word_b_info.label.clone(), jaccard));
                }
                if jaccard.gt(&max_so_far) {
                    r = Some((word_b_info.label.clone(), jaccard));
                    max_so_far = jaccard;
                    last_win_ago = 0;
                } else {
                    last_win_ago += 1;
                }
            }
        }
        // 86 μs
        r
    }
}
