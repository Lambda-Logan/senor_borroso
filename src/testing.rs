use std::cmp::max;
use std::time::Instant;

use hnsw::{Hnsw, Searcher};
use levenshtein::levenshtein;
use rand_pcg::Pcg64;
use space::Neighbor;

use crate::ftzrs::*;
use crate::fuzzyindex::*;
use crate::fuzzypoint::*;

#[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct TrainingAtom<T> {
    pub correct: T,
    pub typo: T,
}

pub trait Testable {
    type Item: Eq + Clone;
    fn fix_error(&mut self, typo: &Self::Item) -> Self::Item;
    fn is_false_negative(&self, ta: &TrainingAtom<Self::Item>, a: &Self::Item) -> bool {
        false
    }

    fn sim(&self, a: &Self::Item, b: &Self::Item) -> f64;

    fn run(&mut self, data: Vec<TrainingAtom<Self::Item>>) {
        let mut errors: Vec<(TrainingAtom<Self::Item>, Self::Item)> =
            Vec::with_capacity(data.len());
        let mut data2 = data.clone();
        let start = Instant::now();
        for atom in data2.drain(..) {
            let guess = self.fix_error(&atom.typo);
            if guess != atom.correct {
                errors.push((atom, guess));
            }
        }
        let elapsed = start.elapsed().as_micros();
        let mut false_negatives = 0;
        let n_errors = errors.len();
        let mut lookup_errors = 0;
        let mut sim_errors = 0;

        for (atom, guess) in errors.iter() {
            if self.is_false_negative(&atom, &guess) {
                false_negatives += 1;
            } else {
                // if the similarity m honestly thinks
                // that its wrong answer is better than the correct one,
                // why should we blame the look process itself??
                if self.sim(&atom.typo, &guess) > self.sim(&atom.typo, &atom.correct) {
                    sim_errors += 1;
                } else {
                    lookup_errors += 1;
                }
            }
        }
        let n_correct = (data.len() - n_errors + false_negatives) as f32;
        //let n_wrong = (data.len() as f32) - n_correct;
        println!(
            "time per lookup: {:?} μs",
            elapsed / max(data.len() as u128, 1)
        );
        println!("number of data points: {:?}", data.len());
        println!("accuracy: {:?}", (n_correct / (data.len() as f32)));
        let true_errors = (sim_errors + lookup_errors) as f32;
        println!(
            "errors from similarity metric: {:?}%",
            100.0 * sim_errors as f32 / true_errors
        );
        println!(
            "errors from lookup process: {:?}%",
            100.0 * lookup_errors as f32 / true_errors
        );

        println!(
            "average comparisons per lookup: {:?}",
            self.average_comparisons(data.iter().take(10_000).map(|x| x.clone()))
        );
    }

    fn average_comparisons<Data>(&mut self, data: Data) -> f32
    where
        Data: Iterator<Item = TrainingAtom<Self::Item>>,
    {
        f32::NAN
    }
}

pub struct FuzzyIndexTest<Ftzr: CanGram, P: FuzzyPoint, M: Metric<P>> {
    pub lookup: FuzzyIndex<String, Ftzr, P>,
    pub params: SearchParams<M>,
    sp: ScratchPad,
}

impl<Ftzr: CanGram, P: FuzzyPoint, M: Metric<P>> FuzzyIndexTest<Ftzr, P, M> {
    pub fn new(lookup: FuzzyIndex<String, Ftzr, P>, params: SearchParams<M>) -> Self {
        FuzzyIndexTest {
            lookup: lookup,
            params: params,
            sp: ScratchPad::new(),
        }
    }
}

impl<Ftzr: CanGram, P: FuzzyPoint, M: Metric<P> + Clone> Testable for FuzzyIndexTest<Ftzr, P, M> {
    type Item = String;
    fn fix_error(&mut self, typo: &String) -> String {
        let pair = self
            .lookup
            .best_match(typo.to_owned(), &mut self.sp, &self.params);
        pair.unwrap_or(("".to_owned(), 0.0)).0
    }
    fn is_false_negative(&self, ta: &TrainingAtom<String>, a: &String) -> bool {
        levenshtein(&ta.correct, &ta.typo) >= levenshtein(&ta.correct, a)
    }
    fn sim(&self, a: &String, b: &String) -> f64 {
        let p_a = P::made_from(a, &self.lookup.ftzr);
        let p_b = P::made_from(b, &self.lookup.ftzr);
        self.params.metric.sim(&p_a, &p_b)
    }

    fn average_comparisons<Data>(&mut self, data: Data) -> f32
    where
        Data: Iterator<Item = TrainingAtom<Self::Item>>,
    {
        let params = self
            .params
            .with_metric(Counted::new(self.params.metric.clone()));
        let mut total = 1;
        for ta in data {
            self.lookup
                .best_match(ta.typo.clone(), &mut self.sp, &params);
            total += 1;
        }
        params.metric.tally.get() as f32 / (total as f32)
    }
}

pub struct HnswTester<
    Ftzr: CanGram,
    Mtrc: Metric<Point>,
    Point: FuzzyPoint,
    const M: usize,
    const M0: usize,
> {
    hnsw: Hnsw<HnswPoint<Mtrc, Point>, Pcg64, M, M0>,
    searcher: Searcher,
    ftzr: Ftzr,
    metric: Mtrc,
    ef: usize,
    words: Vec<String>,
}

impl<
        Ftzr: CanGram,
        Mtrc: Clone + Metric<Point>,
        Point: FuzzyPoint,
        const M: usize,
        const M0: usize,
    > HnswTester<Ftzr, Mtrc, Point, M, M0>
{
    pub fn new(ftzr: Ftzr, m: Mtrc, ef: usize, words: Vec<String>) -> Self {
        let mut hn = Hnsw::new();
        let mut searcher = Searcher::new();
        for word in words.iter() {
            let point = Point::made_from(word, &ftzr);
            hn.insert(HnswPoint::new(m.clone(), point), &mut searcher);
        }
        HnswTester {
            hnsw: hn,
            searcher: searcher,
            ftzr: ftzr,
            metric: m,
            ef: ef,
            words: words,
        }
    }
}

impl<
        Ftzr: CanGram,
        Mtrc: Clone + Metric<Point>,
        Point: FuzzyPoint,
        const M: usize,
        const M0: usize,
    > Testable for HnswTester<Ftzr, Mtrc, Point, M, M0>
{
    type Item = String;
    fn fix_error(&mut self, typo: &String) -> String {
        let mut neighbors = [Neighbor::invalid(); 1];
        let point = Point::made_from(typo, &self.ftzr);

        //when used for benchmarking, the metric.clone() below
        //will only ever be a copy of a unitary type and will be practically instant
        let n = self.hnsw.nearest(
            &HnswPoint::new(self.metric.clone(), point),
            self.ef,
            &mut self.searcher,
            &mut neighbors,
        )[0];

        self.words[n.index].to_owned()
    }

    fn is_false_negative(&self, ta: &TrainingAtom<String>, a: &String) -> bool {
        levenshtein(&ta.correct, &ta.typo) >= levenshtein(&ta.correct, a)
    }

    fn average_comparisons<Data>(&mut self, data: Data) -> f32
    where
        Data: Iterator<Item = TrainingAtom<Self::Item>>,
    {
        let data_vec: Vec<_> = Iterator::collect(data);
        let mut hn: Hnsw<HnswPoint<Counted<Mtrc>, Point>, Pcg64, M, M0> = Hnsw::new();
        let counted: Counted<Mtrc> = Counted::new(self.metric.clone());
        for ta in data_vec.iter() {
            let counted_point =
                HnswPoint::new(counted.clone(), Point::made_from(&ta.correct, &self.ftzr));
            hn.insert(counted_point, &mut self.searcher);
        }

        let mut total = 0;
        counted.tally.set(0);

        for ta in data_vec.iter() {
            let point = HnswPoint::new(counted.clone(), Point::made_from(&ta.typo, &self.ftzr));
            let mut neighbors = [Neighbor::invalid(); 1];
            let n = hn.nearest(&point, self.ef, &mut self.searcher, &mut neighbors)[0];
            total += 1;
        }
        counted.tally.get() as f32 / (total as f32)
    }

    fn sim(&self, a: &String, b: &String) -> f64 {
        let p_a = Point::made_from(a, &self.ftzr);
        let p_b = Point::made_from(b, &self.ftzr);
        self.metric.sim(&p_a, &p_b)
    }
}
