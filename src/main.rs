//#![allow(warnings)]
use std::collections::HashSet;
use std::fs::File;
use std::io::prelude::*;
<<<<<<< HEAD
use std::iter::FromIterator;
=======
>>>>>>> tok_feq_sorted_hash
use std::path::Path;
use std::str;
<<<<<<< HEAD
use std::time::Instant;

mod stacksort;
use stacksort::*;

#[allow(missing_copy_implementations)]
struct FnvHasher(u64);

impl Default for FnvHasher {
    #[inline]
    fn default() -> FnvHasher {
        FnvHasher(0xcbf29ce484222325)
    }
}

impl Hasher for FnvHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let FnvHasher(mut hash) = *self;

        for byte in bytes.iter() {
            hash = hash ^ (*byte as u64);
            hash = hash.wrapping_mul(0x100000001b3);
        }

        *self = FnvHasher(hash);
    }
}

/// A builder for default FNV hashers.
type FnvBuildHasher = BuildHasherDefault<FnvHasher>;
type FnvHashMap<K, V> = HashMap<K, V, FnvBuildHasher>;

fn shuffle<T: Hash + Copy>(items: &[T]) -> Vec<T> {
    let make_hash = |t: T| {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    };
    let mut xs: Vec<_> = Iterator::collect(items.iter().map(|t| Entry {
        id: make_hash(*t),
        entry: t,
    }));
    xs.sort();
    Iterator::collect(xs.into_iter().map(|e| *e.entry))
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq)]
struct Tok(u8, [u8; 2]);

impl fmt::Debug for Tok {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a_val = &[self.0];
        let a = str::from_utf8(a_val).unwrap();
        let bc = str::from_utf8(&self.1).unwrap();
        f.debug_tuple("Tok").field(&a).field(&bc).finish()
    }
}
#[derive(Copy, Clone, Debug)]
struct Entry<Id, T> {
    id: Id,
    entry: T,
}

fn get_entry<'a, Id: Ord, T>(
    entries: &'a [Entry<Id, T>],
    t: &Id,
) -> Option<(usize, &'a Entry<Id, T>)> {
    unsafe {
        match entries.binary_search_by(|e| e.id.cmp(t)) {
            Ok(idx) => Some((idx, entries.get_unchecked(idx))),
            _ => None,
        }
    }
}

impl<T, Id: PartialEq> PartialEq for Entry<Id, T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T, Id: Eq> Eq for Entry<Id, T> {}

impl<T, Id: PartialOrd> PartialOrd for Entry<Id, T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl<T, Id: Ord> Ord for Entry<Id, T> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

fn update_gram<F>(s: &[u8], updt: &mut F)
where
    F: FnMut(Tok) -> (),
{
    let l = s.len();
    for c in 1..l {
        let b = c - 1;
        let _a = cmp::max(0, (b as isize) - 3) as usize;
        for a in _a..b {
            updt(Tok(s[a], [s[b], s[c]]));
        }
    }
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
struct FuzzyEntry {
    string: String,
    bow: Vec<Tok>, //FnvHashMap<Tok, u8>,
}

impl FuzzyEntry {
    fn new(s: String) -> FuzzyEntry {
        let mut vbow: Vec<Tok> = Vec::with_capacity(8);
        {
            let mut updt = |t: Tok| vbow.push(t);
            update_gram(&s.as_bytes(), &mut updt);
        }
        vbow.sort();
        let r = FuzzyEntry {
            string: s,
            bow: vbow,
        };
        //println!("\n{:?}\n", r.bow.len());
        r
    }

    #[inline(always)]
    fn sim(&self, other: &Self) -> f64 {
        let mut n = 0;
        let mut d = 0;
        for x in DualIter::new(self.bow.iter(), other.bow.iter()) {
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
        (n as f64 / d as f64) //.sqrt()
    }
}

#[derive(Debug, Clone)]
struct FreqCounter<T> {
    to_sort: [T; 250],
    freqs: [(T, u8); 250],
}

impl<T: Ord + fmt::Debug + Default + Copy> FreqCounter<T> {
    fn new() -> FreqCounter<T> {
        FreqCounter {
            to_sort: [Default::default(); 250],
            freqs: [Default::default(); 250],
        }
    }

    fn frequencies<'a, Items>(&'a mut self, items: Items) -> &'a mut [(T, u8)]
    where
        Items: Iterator<Item = T>,
    {
        let mut ts = 0;
        for item in items.into_iter().take(250) {
            self.to_sort[ts] = item;
            ts += 1;
        }
        self.to_sort[0..ts].sort();

        //println!("{:?}", self.to_sort);
        let mut f = 0;
        for item in &self.to_sort[..ts] {
            //println!("\nitem: {:?}", item);
            //println!("{:?}", &self.freqs[..10]);
            //if f == 0 {
            //    self.freqs[0] = (*item, 1);
            //}
            if self.freqs[f].0 == *item {
                //TODO and f > 0
                let (t, n) = self.freqs[f];
                self.freqs[f] = (t, n + 1);
            } else {
                f += 1;
                self.freqs[f] = (*item, 1);
            }
            //println!("{:?}", &self.freqs[..10]);
        }
        //println!("{:?}", &self.freqs[..16]);
        &mut self.freqs[1..f + 1]
    }
}
=======
use unidecode::unidecode;

mod testing;
use testing::{FuzzyIndexTest, HnswTester, Testable, TrainingAtom};

mod feat;
//use feat::{
//    book_ends, n_gram, skipgram, AnonFtzr, BookEndsFtzr, CanGram, DefaultAscii, DefaultUnicode,
//    Doc, DocFtzr, EmptyFtzr, FeatEntry, Featurizer, FuzzyEntry, MultiFtzr, SkipScheme,
//};

mod dualiter;
use dualiter::*;
>>>>>>> tok_feq_sorted_hash

mod ftzrs;
use ftzrs::{book_ends, n_gram, skipgram, CanGram, EmptyFtzr, MultiFtzr};

mod fuzzyindex;
use fuzzyindex::{FuzzyIndex, SearchParams};

mod fuzzypoint;
use fuzzypoint::{Counted, FuzzyPoint, Hamming, Jaccard, Labeled, Metric, SimHash, SimplePoint};

mod pointfactories;
use pointfactories::{HasFeatures, HasName};

mod utils;
use utils::{get_entry, open_lexicon, rec_rev_str, shuffle, Entry};
//fn test_index<Ftzr: CanGram, Point: FuzzyPoint>(lookup: &FuzzyIndex<&String, Ftzr, Point>) {

fn induce_typo(word: &str) -> String {
    let mut messed_up: String = "A".to_owned();
    messed_up.push_str(&unidecode(&word));

<<<<<<< HEAD
        too_common.sort();

        let mut sort_slate: &mut Vec<u32> = &mut Vec::new();
        for e in self.toks.iter_mut() {
            let mut words = &mut e.entry;
            sort_slate.extend(words.iter());
            //words.extend(sort_slate.into_iter().filter(|word: u32|{get_entry(&too_common, &(**word as usize, e.id)).is_none()}))
            words.clear();
            for word_idx in sort_slate.iter() {
                if get_entry(&too_common, &(*word_idx as usize, e.id)).is_none() {
                    words.push(*word_idx);
                }
            }
            sort_slate.clear();
        }
    }

    fn lookup(&self, word: String, word_fc: &mut FreqCounter<u32>) -> Option<(String, f64)> {
        let word_info = FuzzyEntry::new(word);
        /// 22 µs
        let mut freq_bow = Vec::with_capacity(word_info.bow.len());
        //  24 µs
        freq_bow.extend(word_info.bow.iter().filter_map(|a| {
            let r = get_entry(&self.toks, &a).map(|(idx, e)| Entry {
                id: e.entry.len(),
                entry: (idx, a),
            });
            r
        }));
        //  73 µs
        freq_bow.sort();
        //  80 µs
        let n_toks = word_info.bow.len() as f64;
        let mut r: Option<(String, f64)> = None;
        let mut max_so_far: f64 = 0.0;

        let word_idxs = freq_bow
            .into_iter()
            .take(5)
            .map(|e| (self.toks[e.entry.0 as usize]).entry.iter())
            .flatten()
            .map(|x| *x);
        //.take(256);
        //  80 µs
        let words_freqs: &mut [(u32, u8)] = word_fc.frequencies(word_idxs);
        //  183 µs
        words_freqs.sort_unstable_by(|a, b| (b.1).cmp(&a.1));
        //  217 µs
        let mut last_win_ago = 0;

        for (word_idx, _) in words_freqs.iter().take(20) {
            if last_win_ago > 12 {
                break;
            }
            let word_b_info: &FuzzyEntry = &self.words[*word_idx as usize];
            let jaccard = word_info.sim(word_b_info); //((freq * 2) as f64) / (n_toks + n_b_toks);
            if jaccard > 0.8 {
                return Some((word_b_info.string.clone(), jaccard));
            }
            if jaccard.gt(&max_so_far) {
                r = Some((word_b_info.string.clone(), jaccard));
                max_so_far = jaccard;
                last_win_ago = 0;
            } else {
                last_win_ago += 1;
            }
        }

        r
    }
=======
    if word.len() > 4 {
        messed_up.insert_str(4, "E");
    };
    messed_up
>>>>>>> tok_feq_sorted_hash
}

#[macro_export]
macro_rules! featurizers {
    () => {
        (EmptyFtzr)
    };
<<<<<<< HEAD

    // Read the file contents into a string, returns `io::Result<usize>`
    let mut s = String::new();
    match file.read_to_string(&mut s) {
        Err(why) => panic!("couldn't read {}: {}", display, why),
        Ok(_) => {} //print!("{} contains:\n{}", display, s),
    }
    s = s.to_uppercase();
    let mut words = HashSet::new();
    for line in s.lines() {
        let line_words: Vec<_> = line.split("|").collect();
        if line_words.len() > 2 {
            if true {
                //////////////////////////////////////////////////////////////////////////////////////
                words.insert(line_words[0].to_owned());
            } else if line_words[1] == "MI" {
                words.insert(line_words[0].to_owned());
            }
        }
    }

    let lookup = FuzzyIndex::new(words.iter().map(|a| a.to_owned()));
    let mut wins = 0;
    let mut n_cities_done: usize = 0;
    let n_cities = 10000;
    let mut fc = FreqCounter::new();
    //println!("{:?}", fc.frequencies([77; 249].iter().map(|d| { *d })));

    let fe: Vec<_> = Iterator::collect(words.iter().map(|w| FuzzyEntry::new(w.to_owned())));
    let start = Instant::now();
    if false {
        for c1 in fe.iter().take(1000) {
            for c2 in fe.iter().take(1000) {
                c1.sim(c2);
                n_cities_done += 1;
            }
        }
    }

    for city in words.iter() {
        //.take(n_cities) {
        n_cities_done += 1;
        let mut messed_up: String = "A".to_owned();
        messed_up.push_str(city);
        //let e = FuzzyEntry::new(messed_up);
        if true {
            match lookup.lookup(messed_up, &mut fc) {
                Some((r, _)) => {
                    if &r == city {
                        wins += 1;
                    }
                }
                _ => {}
            }
        }
    }
    let elapsed = start.elapsed().as_micros();
    println!(
        "TIME EACH: {:?}",
        elapsed / cmp::max(n_cities_done as u128, 1)
    );
    println!(
        "{:?}",
        (wins, n_cities_done, (wins as f32) / (n_cities_done as f32))
    );

    let l = vec![0, 0, 4, 14, 20, 20, 24];
    let r = vec![4, 10, 10, 14, 24];
    for i in DualIter::new(l.iter(), r.iter()) {
        //println!("{:?}", i);
    }
    println!(
        "{:?}",
        //FuzzyEntry::new("GRAN RAPIDS".to_owned()).sim(&FuzzyEntry::new("GRAND RAPIDS".to_owned()))
        //lookup.lookup("RAPIDS".to_owned())
        shuffle(&[1, 2, 3, 4, 5, 6, 7])
    )
=======
    ($a:expr $(, $tail:expr)*) => {{
        MultiFtzr {
            a: $a,
            b: featurizers!($($tail), *),
        }
    }};
>>>>>>> tok_feq_sorted_hash
}

//impl<T: Ord> MetricPoint for FuzzyEntry<T> {
//    fn distance(&self, rhs: &Self) -> u64 {
//        //space::f64_metric(1.0 - self.sim(rhs))
//        self.ham_dist(rhs)
//    }
//}
fn main() {}
