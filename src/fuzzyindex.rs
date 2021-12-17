use std::cmp;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::marker::PhantomData;

use crate::clustering::*;
use crate::dualiter::Uniq;
use crate::ftzrs::*;
use crate::fuzzypoint::*;
use crate::hasfeatures::*;
use crate::metrics::Metric;
use crate::testing::StatsPad;
use crate::utils::shuffle;
use crate::utils::{get_entry, Entry};

/////////////////////
/// A struct for finding the frequency of elements in an iter.
/// This is usually faster than a HashMap
/// Use with '1' per element for counting purposes.
/// ////////////////
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

    fn frequencies<'a, Items>(&'a mut self, items: Items) -> (usize, &'a mut Vec<(T, u32)>)
    where
        Items: Iterator<Item = (T, u32)>,
    {
        self.to_sort.clear();
        self.freqs.clear();

        self.to_sort.extend(items);
        let length = self.to_sort.len();
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
        (length, &mut self.freqs)
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
    pub(crate) stats: Option<StatsPad<usize>>,
}
impl ScratchPad {
    pub fn new() -> Self {
        ScratchPad {
            fc: FreqCounterWith::new(),
            freq_bow: Vec::new(),
            stats: None,
        }
    }
}

impl Default for ScratchPad {
    fn default() -> Self {
        ScratchPad::new()
    }
}
/////////////////////
/// The runtime search parameters for a lookup.
///
/// The 'M' parameters allows different comparison metrics to be used dynamically at runtime.
/// 'depth' is how many features of the input point are examined. This is the most important parameter and has the largest effect on accuracy vs speed.
/// 'breadth' is the max number of other points found PER FEATURE of 'depth'
/// 'use_best_after: n' will stop after not finding a new closest point after any 'n' comparisons.
/// 'return_if_gt' should be a number 0-100 representing similarity 0.0-1.0. For example, if 'return_if_gt' is 80, the search will stop after finding a point with a similarity of 0.8 or higher.
/// 'max_comparisons' is the maximum number of comparisons to be made on a given lookup.
/// # Examples
/// ```
///let params1 = SearchParams {
///    metric: Hamming,
///    depth: 50,
///    breadth: 100,
///    max_comparisons: 500,
///    return_if_gt: 90,
///    use_best_after: 144,
///};
///
///let params2 = SearchParams {
///    metric: Jaccard,
///    depth: 25,
///    breadth: 25,
///    max_comparisons: 100,
///    return_if_gt: 75,
///    use_best_after: 15,
///};
/// ```
/// ////////////////
#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct SearchParams<M> {
    pub metric: M,
    pub depth: usize,
    pub breadth: usize,
    pub use_best_after: u32,
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
    ///                  use_best_after: 20,
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
            use_best_after: self.use_best_after,
        }
    }
}
/////////////////////
/// A lookup returns 0 or more of 'Neighbor'.
///
/// # Examples
/// ```
/// for neighbor in fuzzyindex.neighbors_of(&"MY LOOKUP", &mut scratchpad, &params) {
///     println("label: {:?}", neighbor.label);
///     println("sim: {:?}\n", neighbor.similarity);
/// }
/// ```
/// ////////////////
#[derive(Copy, Clone, Debug)]
pub struct Neighbor<Label> {
    pub label: Label,
    pub similarity: f64,
}

impl<Label> PartialOrd for Neighbor<Label> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.similarity.partial_cmp(&other.similarity)
    }
}

impl<Label> PartialEq for Neighbor<Label> {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

//////////////
/// The iterator over the nearest neighbors of a point.
/// Returned by 'fuzzyindex.neighbors_of' or 'fuzzyindex.neighbors_of_point'
/// //////////
pub struct Neighborhood<'idx, 'sp, Origin, Tok, Label, Ftzr, Point, Mtrc: Metric<Point>> {
    index: &'idx FuzzySearchIndex<Origin, Tok, Label, Ftzr, Point>,
    params: SearchParams<Mtrc>,
    point: Point,
    items: std::slice::Iter<'sp, (u32, u32)>,
    best_so_far: Option<Neighbor<&'idx Label>>,
    best_ago: u32,
    return_if_gt: f64,
    n_left: usize,
}

impl<'idx, 'sp, Origin, Tok, Label, Ftzr, Point, Mtrc: Metric<Point>>
    Neighborhood<'idx, 'sp, Origin, Tok, Label, Ftzr, Point, Mtrc>
{
    pub(crate) fn new(
        index: &'idx FuzzySearchIndex<Origin, Tok, Label, Ftzr, Point>,
        params: SearchParams<Mtrc>,
        point: Point,
        items: &'sp [(u32, u32)],
    ) -> Self {
        Neighborhood {
            index: index,
            n_left: params.max_comparisons,
            return_if_gt: params.return_if_gt as f64 / 100.0,
            params: params,
            point: point,
            items: items.iter(),
            best_so_far: None,
            best_ago: 0,
        }
    }
    //////////////////////
    /// Returns the nearest neighbor found in a 'Neighborhood'.
    /// # EXAMPLES
    /// ```
    /// let nn: Option<Neighbor<_>> = fuzzyindex
    ///                     .neighbors_of(&"MY LOOKUP", &mut scratchpad, &params)
    ///                     .nearest();
    /// ```
    pub fn nearest(mut self) -> Option<Neighbor<&'idx Label>> {
        let mut i = self.next();
        while i.is_some() {
            i = self.next();
        }
        self.best_so_far
    }
}

impl<'idx, 'sp, Origin, Tok, Label, Ftzr, Point, Mtrc: Metric<Point>> Iterator
    for Neighborhood<'idx, 'sp, Origin, Tok, Label, Ftzr, Point, Mtrc>
{
    type Item = Neighbor<&'idx Label>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.n_left == 0 {
            return None;
        } else {
            self.n_left -= 1;
        }

        if self.best_ago > self.params.use_best_after {
            return None;
        } else {
            self.best_ago += 1; // TODO don't increment 'best_ago' if the similarity == 1.0
        }

        let n = self.items.next();
        match n {
            None => self.best_so_far,
            Some((point_b_idx, _)) => {
                if self.best_ago > self.params.use_best_after {
                    return self.best_so_far;
                }
                let point_b = &self.index.points[*point_b_idx as usize];
                let sim = self.params.metric.sim(&self.point, &point_b.point);

                if sim > self.return_if_gt {
                    self.n_left = 0;
                    return Some(Neighbor {
                        label: &point_b.label,
                        similarity: sim,
                    });
                }
                match self.best_so_far {
                    None => {
                        self.best_so_far = Some(Neighbor {
                            label: &point_b.label,
                            similarity: sim,
                        });
                        Some(Neighbor {
                            label: &point_b.label,
                            similarity: sim,
                        })
                    }
                    Some(best_so_far) => {
                        if sim.gt(&best_so_far.similarity) {
                            let r = Some(Neighbor {
                                label: &point_b.label,
                                similarity: sim,
                            });
                            self.best_so_far = r;
                            self.best_ago = 0;
                        };

                        Some(Neighbor {
                            label: &point_b.label,
                            similarity: sim,
                        })
                    }
                }
            }
        }
    }
}

//////////
/// This is a macro used to record the runtime stats in the 'stats' field of ScratchPad
/// ////////
macro_rules! record_stats (
    ($sp_name:ident, $action:ident, $val:expr) => {
        match $sp_name.stats.as_mut() {
            None=>(),
            Some(statspad)=>{
                statspad.$action += $val;
            }
        };
    };
);

//////////////////////
/// The type of a fuzzy index.
/// # EXAMPLES
///```
///let ftzr = n_gram(2);
///
///let coral_names = vec![
///    "acropora tenuis",
///    "acropora torihalimeda",
///    "acropora tortuosa",
///    "acropora turaki",
///    "acropora valenciennesi",
///    "acropora valida",
///    "acropora variolosa",
///];
///
///let mut scratchpad = Default::default();
///
///let coral_name_index: FuzzyIndex<&str, _, SimplePoint> =
///    FuzzyIndex::new(ftzr, coral_names.into_iter());
///
///let typo = "acorpora torehalemida";
///
///let params = SearchParams {
///    metric: Jaccard,
///    depth: 20,
///    breadth: 10,
///    max_comparisons: 5,
///    return_if_gt: 70,
///    use_best_after: 5,
///};
///
///let correct = coral_name_index
///    .neighbors_of(typo, &mut scratchpad, &params)
///    .nearest()
///    .unwrap();
///
///println!("{:?}", correct);
/// // >> Neighbor { label: "acropora torihalimeda", similarity: 0.3793103448275862 }
/// ```
#[derive(Clone, Debug)]
pub struct FuzzySearchIndex<Origin, Tok, Label, Ftzr, Point> {
    pub ftzr: Ftzr,
    pub(crate) points: Vec<Labeled<Point, Label>>,
    pub(crate) all_feats: Vec<Entry<Feature, Vec<u32>>>,
    _marker: PhantomData<(Origin, Tok)>,
}

pub type FuzzyIndex<Origin: HasLabel + HasFeatures, Ftzr, Point: FuzzyPoint> = FuzzySearchIndex<
    Origin,
    <Origin as HasFeatures>::Tok,
    <Origin as HasLabel>::Label,
    Ftzr,
    Point,
>;

impl<Origin, Tok, Label, Ftzr, Point> FuzzySearchIndex<Origin, Tok, Label, Ftzr, Point>
where
    Tok: Sized + Hash + Debug,
    Origin: HasFeatures<Tok = Tok> + HasLabel<Label = Label> + Eq + Hash,
    Point: FuzzyPoint + FromFeatures,
    Ftzr: CanGram,
    Label: Clone,
{
    /////////////////////
    /// A fuzzy index is created from an iterable of sequences that can be featurized.
    /// the arguments to 'new' are a featurizer and the iterable of sequences.
    /// # Examples
    /// ```
    ///let codes: Vec<[u8; 4]> = vec![
    ///    [1, 2, 3, 4],
    ///    [2, 3, 4, 5],
    ///    [7, 2, 2, 3],
    ///    // + thousands of other rows
    ///    [1, 2, 7, 2],
    ///];
    ///
    ///let labeled_codes = codes.into_iter().enumerate().map(Labeled::from_tuple);
    ///
    ///let fuzzy_code_index: FuzzyIndex<Labeled<[u8; 4], usize>, _, SimplePoint> =
    ///    FuzzyIndex::new(n_gram(2), labeled_codes);
    /// ```
    /////////////////////
    pub fn new<W: Iterator<Item = Origin>>(ftzr: Ftzr, sequences: W) -> Self {
        let words_set: HashSet<Origin> = Iterator::collect(sequences);
        // sort points set???
        let mut points: Vec<Labeled<Point, Label>> =
            Iterator::collect(words_set.into_iter().map(|x| Labeled {
                label: x.label(),
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
    ///////////////////
    /// returns the number of input sequences that contain the given feature
    ///////////////////
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

    ////////////////////
    /// Takes a sequence that 'HasFeatures' and returns a iter of neigbors. (a Neighborhood).
    /// If you have already extracted a point from the features of a sequence using FromFeatures::made_from(&sequence, &ftzr), then use 'index.neighbors_of_point'
    /// # Examples
    /// ```
    /// let my_doc = "one fish two fish red fish blue fish".split();
    /// for neighbor in fuzzy_doc_index.neighbors_of(&my_doc, &mut scratchpad, &params) {
    ///     println("label: {:?}", neighbor.label);
    ///     println("sim: {:?}\n", neighbor.similarity);
    /// }
    /// ```
    //////////////
    pub fn neighbors_of<'idx, 'sp, M: Clone + Metric<Point>>(
        &'idx self,
        tokens: Origin,
        sp: &'sp mut ScratchPad,
        params: &SearchParams<M>,
    ) -> Neighborhood<'idx, 'sp, Origin, Tok, Label, Ftzr, Point, M> {
        record_stats!(sp, called, 1);
        let point = Point::made_from(&tokens, &self.ftzr);
        self.neighbors_of_point(point, sp, &params)
    }

    pub(crate) fn adj_matrix(&self) -> AdjacencyMatric<SimplePoint> {
        let mut v: Vec<Entry<Feature, SimplePoint>> = Vec::with_capacity(self.all_feats.len());
        for e in self.all_feats.iter() {
            let feat = e.id;
            v.push(Entry {
                id: feat,
                entry: SimplePoint::new(Vec::new()),
            });
        }
        for labeled_point in self.points.iter() {
            for feat_a in labeled_point.point.get_sorted_features().iter() {
                for feat_b in labeled_point.point.get_sorted_features().iter() {
                    if feat_a != feat_b {
                        let idx = get_entry(&v, feat_a).unwrap().0;
                        v[idx].entry.get_features_mut().push(*feat_b);
                    }
                }
            }
        }
        for e in v.iter_mut() {
            e.entry.get_features_mut().sort();
        }
        v
    }
}

impl<Origin, Tok, Label, Ftzr, Point> FuzzySearchIndex<Origin, Tok, Label, Ftzr, Point>
where
    Tok: Sized + Hash + Debug,
    Point: FuzzyPoint,
    Ftzr: CanGram,
    Origin: HasLabel<Label = Label> + Eq + Hash + HasFeatures,
    Label: Clone,
{
    ////////////////////////////
    /// Similar to 'FuzzyIndex::new', but used if you've already done featurization.
    /// Takes an iter of 'Labeled' points
    ////////////////////////////
    pub fn from_points<W: Iterator<Item = Labeled<Point, Label>>>(
        ftzr: Ftzr,
        sequences: W,
    ) -> Self {
        //let words_set: HashSet<Origin> = Iterator::collect(sequences);
        // sort points set???
        let mut points: Vec<Labeled<Point, Label>> = Iterator::collect(sequences.into_iter());

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

    pub(crate) fn transpose(&self) -> FuzzyIndex<FeatCol, EmptyFtzr, FeatCol> {
        FuzzyIndex::from_points(
            EmptyFtzr,
            self.all_feats.iter().map(|e| Labeled {
                label: e.id.0,
                point: FeatCol {
                    id: e.id.0,
                    feats: Iterator::collect(
                        e.entry.iter().map(|word_idx| Feature(*word_idx as u64)),
                    ),
                    df: e.entry.len(),
                },
            }),
        )
    }

    ////////////////////////
    /// Similar to 'FuzzyIndex::neighbors_of', but used if you've already done featurization.
    /// Takes a point that implements 'FuzzyPoint'
    /// ////////////////////
    pub fn neighbors_of_point<'idx, 'sp, M: Clone + Metric<Point>>(
        &'idx self,
        point: Point,
        sp: &'sp mut ScratchPad,
        params: &SearchParams<M>,
    ) -> Neighborhood<'idx, 'sp, Origin, Tok, Label, Ftzr, Point, M> {
        record_stats!(sp, input_features, point.len());
        sp.freq_bow.clear(); //Vec::with_capacity(point.get_sorted_features().len());
                             // 36 ms
        sp.freq_bow.extend(
            Uniq::new(point.get_sorted_features().iter()).filter_map(|a| {
                let r = get_entry(&self.all_feats, &a).map(|(idx, e)| Entry {
                    id: e.entry.len(),
                    entry: (idx, *a),
                });
                r
            }),
        );
        /*record_stats!(sp, input_features_unique, {
            let mut i = 0;
            for _ in Uniq::new(point.get_sorted_features().iter()) {
                i += 1;
            }
            i
        });*/
        sp.freq_bow.sort();
        // 25 μs
        let n_toks = point.get_sorted_features().len() as f64;
        let mut r: Option<(Label, f64)> = None;
        let mut max_so_far: f64 = 0.0;
        let word_idxs = sp
            .freq_bow
            .iter()
            .take(params.depth) //48
            .map(|e| {
                let tfidf: u32 = 512 / ((e.id as f64).log2() as u32 + 1);
                //record_stats!(sp, inc_nbrs_total, 1);
                (self.all_feats[e.entry.0 as usize])
                    .entry
                    .iter()
                    .take(params.breadth)
                    .map(move |entry| (*entry, tfidf))
            })
            .flatten();

        let (nbrs_total, words_freqs) = sp.fc.frequencies(word_idxs);
        // 47 μs
        words_freqs.sort_unstable_by(|a, b| (b.1).cmp(&a.1));
        record_stats!(sp, nbrs_total, nbrs_total);
        record_stats!(sp, nbrs_unique, words_freqs.len());
        // 51 μs
        let mut last_win_ago = 0;
        let return_if_gt = params.return_if_gt as f64 / 100.0;

        // 86 μs
        Neighborhood::new(&self, params.clone(), point, words_freqs.as_slice())
    }
}
