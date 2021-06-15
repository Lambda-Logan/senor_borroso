#![allow(warnings)]
use std::cmp;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::collections::HashSet;
use std::convert::Into;
use std::default::Default;
use std::fmt;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::io::prelude::*;
use std::iter::FromIterator;
use std::mem;
use std::path::Path;
use std::rc::Rc;
use std::str;
use std::time::Instant;
use unidecode::unidecode;

mod feat;
use feat::{DefaultAscii, DefaultUnicode, FeatEntry, Featurizer, FuzzyEntry};

mod dualiter;
use dualiter::*;

mod utils;
use utils::{get_entry, open_lexicon, rec_rev_str, shuffle, Entry};

//#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq)]
//struct Tok(u8, u8, u8, u8);
//struct Tok(u8, [u8; 2]);

#[derive(Debug, Clone, Default)]
struct FreqCounter<T> {
    to_sort: Vec<T>,
    freqs: Vec<(T, usize)>,
}

impl<T: Ord + fmt::Debug> FreqCounter<T> {
    fn new() -> FreqCounter<T> {
        FreqCounter {
            to_sort: Vec::new(),
            freqs: Vec::new(),
        }
    }

    fn frequencies<'a, Items>(&'a mut self, items: Items) -> &'a mut Vec<(T, usize)>
    where
        Items: Iterator<Item = T>,
    {
        self.to_sort.clear();
        self.freqs.clear();

        self.to_sort.extend(items);
        self.to_sort.sort();
        let mut item_iter = self.to_sort.drain(..);

        for item in item_iter {
            match self.freqs.last_mut() {
                None => {
                    self.freqs.push((item, 1));
                }
                Some(last) => {
                    if last.0 == item {
                        last.1 += 1;
                    } else {
                        self.freqs.push((item, 1));
                    }
                }
            }
        }
        &mut self.freqs
    }
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct SearchParams {
    pub depth: usize,
    pub breadth: usize,
    pub timeout_n: u32,
    pub return_if_gt: u8,
    pub max_comparisons: usize,
}

#[derive(Clone, Debug)]
struct FuzzyIndex<F: Featurizer> {
    ftzr: F,
    words: Vec<FeatEntry<F>>,
    all_feats: Vec<Entry<F::Feat, Vec<u32>>>,
}

impl<F: Featurizer> FuzzyIndex<F> {
    fn new<W: Iterator<Item = F::Origin>>(f: F, wrds: W) -> FuzzyIndex<F> {
        let words_set: HashSet<F::Origin> = Iterator::collect(wrds);
        // sort words set???
        let mut words: Vec<FeatEntry<F>> =
            Iterator::collect(words_set.into_iter().map(|x| f.new(x)));
        words.sort();
        let mut toks_hm: HashMap<F::Feat, Vec<u32>> = HashMap::with_capacity(words.len());
        let empty = Vec::new();
        for (idx, word) in words.iter().enumerate() {
            for tok in word.feats.iter() {
                let v = toks_hm.entry(tok.clone()).or_insert(empty.clone());
                v.push(idx as u32);
            }
        }
        let mut toks = Vec::with_capacity(toks_hm.len());

        for (tok, idxs) in toks_hm.into_iter() {
            //println!("{:?}", idxs.len());
            toks.push(Entry {
                id: tok,
                entry: shuffle(&idxs),
            });
        }
        toks.sort();
        //TODO randomize order of the Vec<u32>'s in toks??
        println!("w t {:?}", (words.len(), toks.len()));

        let mut r = FuzzyIndex {
            ftzr: f,
            words: words,
            all_feats: toks,
        };
        //println!("{:?}", r);
        //r.compress(55);
        //r.compress(34);
        //r.compress(21);
        r.compress(16);
        r.compress(10);
        r
    }

    fn freq_of(&self, f: &F::Feat) -> usize {
        get_entry(&self.all_feats, f)
            .map(|(_, e)| e.entry.len())
            .unwrap_or(0)
    }

    fn compress(&mut self, cuttoff: usize) {
        let mut too_common: Vec<Entry<(usize, F::Feat), ()>> =
            Vec::with_capacity(self.all_feats.len());
        let mut toks_sort: Vec<F::Feat> = Vec::new();
        for (idx, word) in self.words.iter().enumerate() {
            if word.feats.len() > cuttoff {
                toks_sort.extend(word.feats.iter().map(|f| f.clone()));
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
            let mut words = &mut e.entry;
            sort_slate.extend(words.iter());
            //words.extend(sort_slate.into_iter().filter(|word: u32|{get_entry(&too_common, &(**word as usize, e.id)).is_none()}))
            words.clear();
            for word_idx in sort_slate.iter() {
                if get_entry(&too_common, &(*word_idx as usize, e.id.clone())).is_none() {
                    words.push(*word_idx);
                }
            }
            sort_slate.clear();
        }
    }

    fn lookup(
        &self,
        word: &F::Origin,
        word_fc: &mut FreqCounter<u32>,
        params: &SearchParams,
    ) -> Option<(F::Id, f64)> {
        /*let mut word_info = FuzzyEntry::new(word);
        word_info.bow.sort_unstable_by(|a, b| {
            let a_freq = get_entry(&self.toks, &a)
                .map(|e| e.entry.len())
                .unwrap_or(0);
            let b_freq = get_entry(&self.toks, &b)
                .map(|e| e.entry.len())
                .unwrap_or(0);
            a_freq.cmp(&b_freq)
        }); */

        let word_info = self.ftzr.new(word.to_owned());
        let mut freq_bow = Vec::with_capacity(word_info.feats.len());
        // 36 ms
        freq_bow.extend(word_info.feats.iter().filter_map(|a| {
            let r = get_entry(&self.all_feats, &a).map(|(idx, e)| Entry {
                id: e.entry.len(),
                entry: (idx, a),
            });
            r
        }));
        // 64 ms
        freq_bow.sort();
        // 64 ms
        //println!("n_times {:?}", n_times);
        let n_toks = word_info.feats.len() as f64;
        let mut r: Option<(F::Id, f64)> = None;
        let mut max_so_far: f64 = 0.0;
        //println!("{:?}", freqs);
        //println!("{:?}", freq_bow.len());
        //let mut tried = Vec::new();
        //let mut words = Vec::with_capacity(256);
        //for e in freq_bow.into_iter().take(1) {
        //   let (freq, (idx, tok)) = (e.id, e.entry);
        //println!("{:?}", (idx, freq));
        //println!("{:?}", &self.toks[idx as usize].entry.len());
        //   words.extend(self.toks[idx as usize].entry.iter());
        //}
        //let words_len = words.len();
        //let mut words_freqs = word_fc.frequencies(words.drain(..));
        let word_idxs = freq_bow
            .into_iter()
            .take(params.depth) //48
            .map(|e| {
                (self.all_feats[e.entry.0 as usize])
                    .entry
                    .iter()
                    .take(params.breadth)
            })
            .flatten()
            .map(|x| *x);
        //.take(250);
        let words_freqs = word_fc.frequencies(word_idxs);
        // 74 ms
        words_freqs.sort_unstable_by(|a, b| (b.1).cmp(&a.1));
        //println!("{:?}", words_freqs.len());
        // 74 ms
        let mut last_win_ago = 0;
        let return_if_gt = params.return_if_gt as f64 / 100.0;
        for (word_idx, _) in words_freqs.iter().take(params.max_comparisons) {
            if last_win_ago > params.timeout_n {
                break;
            }
            let word_b_info = &self.words[*word_idx as usize];
            let jaccard = word_info.sim(word_b_info); //((freq * 2) as f64) / (n_toks + n_b_toks);
            if jaccard > return_if_gt {
                return Some((word_b_info.id.clone(), jaccard));
            }
            if jaccard.gt(&max_so_far) {
                r = Some((word_b_info.id.clone(), jaccard));
                max_so_far = jaccard;
                last_win_ago = 0;
            } else {
                last_win_ago += 1;
            }
        }
        // 89 ms
        r
    }
}

fn test_index<F: Featurizer<Origin = String, Id = String>>(lookup: &FuzzyIndex<F>) {
    let mut wins = 0;
    let mut n_cities_done: usize = 0;
    //let n_cities = 1000;
    let mut fc = FreqCounter::new();
    //.take(n_cities)
    let _words: Vec<_> = Iterator::collect(lookup.words.iter().map(|w| w.id.to_owned()));
    let words = shuffle(&_words);
    let mut incrct = Vec::with_capacity(100_000);
    let start = Instant::now();
    for city in words {
        //println!("{:?}", city);
        n_cities_done += 1;
        let mut messed_up: String = "A".to_owned();
        messed_up.push_str(&unidecode(&city));
        //println!("A {:?}", city);
        //let mut messed_up: String = city.to_owned();
        for n in 4..8 {
            if city.len() > n {
                if city.is_char_boundary(n) {
                    //println!("E {:?}", city);
                    messed_up.insert_str(n, "E");
                    break;
                }
            }
        }
        //println!("X {:?}", city);
        //println!("{:?}", (city, &messed_up));
        //let e = FuzzyEntry::new(messed_up);
        let params = SearchParams {
            depth: 48,
            breadth: 48,
            max_comparisons: 32,
            return_if_gt: 80,
            timeout_n: 15,
        };
        if true {
            match lookup.lookup(&messed_up, &mut fc, &params) {
                Some((r, _)) => {
                    if r == city {
                        wins += 1;
                    } else {
                        println!("r: {:?}", (&city, &messed_up, &r));
                        incrct.push((city.to_owned(), messed_up.to_owned(), r.to_owned()));
                    }
                }
                _ => {
                    println!("{:?}", city);
                }
            }
        }
    }
    let elapsed = start.elapsed().as_micros();
    println!(
        "TIME EACH: {:?}",
        elapsed / cmp::max(n_cities_done as u128, 1)
    );
    println!(
        "lookup was right: {:?}",
        (wins, n_cities_done, (wins as f32) / (n_cities_done as f32))
    );
    let mut g = 0;
    let mut bad = 0;
    for (a, b, c) in incrct.into_iter() {
        let (right, messed_up, wrng) = (
            DefaultUnicode.new(a),
            DefaultUnicode.new(b),
            DefaultUnicode.new(c),
        );
        if right.sim(&messed_up) < wrng.sim(&messed_up) {
            bad += 1;
        } else {
            g += 1;
        }
    }
    println!(
        "sim was correct: {:?}",
        //FuzzyEntry::new("GRAN RAPIDS".to_owned()).sim(&FuzzyEntry::new("GRAND RAPIDS".to_owned()))
        //lookup.lookup("RAPIDS".to_owned())
        //shuffle(&[1, 2, 3, 4, 5, 6, 7])
        //lookup.lookup(&"GRAN RAID".to_owned(), &mut fc)
        (g as f32 / (bad + g) as f32)
    )
}

fn main() {
    /*let fs = SkipScheme {
        group_a: (0, 2),
        gap: (0, 1),
        group_b: (0, 1),
    };
    fs.run(&['a', 'b', 'c', 'd', 'e', 'f'], &mut |a| {}); */

    //let lookup = FuzzyIndex::new(words);
    //println!("{:?}", lookup.lookup("gra".to_owned()));
    let mut i = 0.0;

    let path = Path::new("/home/logan/Downloads/us_cities_states_counties.csv");

    let display = path.display();

    // Open the path in read-only mode, returns `io::Result<File>`
    let mut file = match File::open(&path) {
        Err(why) => panic!("couldn't open {}: {}", display, why),
        Ok(file) => file,
    };

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
                words.insert(rec_rev_str(line_words[0].to_owned()));
            } else if line_words[1] == "MI" {
                words.insert(line_words[0].to_owned());
            }
        }
    }

    //let lookup = FuzzyIndex::new(DefaultAscii, words.iter().map(|a| a.to_owned()));
    let lookup = FuzzyIndex::new(
        DefaultUnicode,
        open_lexicon(Path::new(
            "/home/logan/Dropbox/USUABLE/en_pl_lexemes/en.txt",
        ))
        .into_iter(),
    );
    //let mut fc = FreqCounter::new();

    test_index(&lookup);
}
