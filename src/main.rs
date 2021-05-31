use std::cmp;
use std::collections::HashMap;
use std::collections::HashSet;
use std::convert::Into;
use std::default::Default;
use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};
use std::iter::FromIterator;
use std::rc::Rc;
use std::str;

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

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
    bow: Vec<Entry<Tok, u8>>, //FnvHashMap<Tok, u8>,
}

struct DualIterX<A, Av, B, Bv> {
    iter_a: A,
    iter_b: B,
    val_a: Option<Av>,
    val_b: Option<Bv>,
}

type DualIter<A: Iterator, B: Iterator> =
    DualIterX<A, <A as Iterator>::Item, B, <B as Iterator>::Item>;

impl FuzzyEntry {
    fn new(s: String) -> FuzzyEntry {
        let mut bow = HashMap::with_capacity_and_hasher(s.len() * 8, FnvBuildHasher::default());
        {
            let mut updt = |t: Tok| {
                let n: u8 = *bow.get(&t).unwrap_or(&0);
                bow.insert(t, n + 1);
            };
            update_gram(&s.as_bytes(), &mut updt);
        }
        let mut vbow: Vec<Entry<Tok, u8>> = Vec::with_capacity(bow.len());
        for (tok, freq) in bow.into_iter() {
            vbow.push(Entry {
                id: tok,
                entry: freq,
            });
        }
        vbow.sort();

        let r = FuzzyEntry {
            string: s,
            bow: vbow,
        };
        //println!("\n{:?}\n", r.bow.len());
        r
    }

    /*
    #[inline(always)]
    fn sim(&self, other: &Self) -> f64 {
        if self.bow.len() == 0 {
            return 0.0;
        }
        let mut n = 0.0;
        let mut d = 0.0;
        for (tok, freq_a) in &self.bow {
            let freq_b = other.bow.get(&tok).unwrap_or(&0);
            n += (*cmp::min(freq_a, freq_b) as f64);
            d += (*cmp::max(freq_a, freq_b) as f64);
        }
        for (tok, freq) in &other.bow {
            if !self.bow.contains_key(&tok) {
                d += (*freq as f64);
            }
        }
        (n / d).sqrt()
    }*/
}
#[derive(Clone, Debug)]
struct FuzzyLookup {
    words: Vec<FuzzyEntry>,
    toks: Vec<Entry<Tok, Vec<u32>>>,
}

impl FuzzyLookup {
    fn new<W: Iterator<Item = String>>(wrds: W) -> FuzzyLookup {
        let words_set: HashSet<String> = Iterator::collect(wrds);
        // sort words set???
        let mut words: Vec<FuzzyEntry> =
            Iterator::collect(words_set.into_iter().map(FuzzyEntry::new));
        words.sort();
        let mut toks_hm: HashMap<Tok, Vec<u32>> = HashMap::with_capacity(words.len());
        let empty = Vec::new();
        for (idx, word) in words.iter().enumerate() {
            for t_e in word.bow.iter() {
                let v = toks_hm.entry(t_e.id).or_insert(empty.clone());
                v.push(idx as u32);
            }
        }
        let mut toks = Vec::with_capacity(toks_hm.len());

        for (tok, idxs) in toks_hm.into_iter() {
            //println!("{:?}", idxs.len());
            toks.push(Entry {
                id: tok,
                entry: idxs,
            });
        }
        toks.sort();
        //TODO randomize order of the Vec<u32>'s in toks??
        println!("w t {:?}", (words.len(), toks.len()));
        let r = FuzzyLookup {
            words: words,
            toks: toks,
        };
        //println!("{:?}", r);
        r
    }

    fn lookup(&self, word: String) -> Option<(String, f64)> {
        let word_info = FuzzyEntry::new(word);
        //println!("{:?}", word_info);
        let mut freqs = HashMap::new(); //with_hasher(FnvHasher::default());
        for t_f in word_info.bow.iter() {
            match self.toks.binary_search(&Entry {
                id: t_f.id,
                entry: Vec::new(),
            }) {
                //TODO don't allocate a new vec each time
                Ok(idx) => {
                    //println!("idx {:?}", idx);
                    for word_idx in self.toks[idx].entry.iter() {
                        *freqs.entry(word_idx).or_insert(0) += 1;
                    }
                }
                _ => {}
            }
        }
        let n_toks = word_info.bow.len() as f64;
        let mut r: Option<(String, f64)> = None;
        let mut max_so_far: f64 = 0.0;
        //println!("{:?}", freqs);
        for (idx, freq) in freqs.into_iter() {
            //println!("{:?}", (idx, freq));
            let word_b_info = &self.words[*idx as usize];
            let n_b_toks = word_b_info.bow.len() as f64;
            let jaccard = ((freq * 2) as f64) / (n_toks + n_b_toks);
            if jaccard.gt(&max_so_far) {
                r = Some((word_b_info.string.clone(), jaccard));
                max_so_far = jaccard;
            }
        }
        r
    }
}

fn main() {
    let words = vec![
        //"zabc".to_owned(),
        //"ab cd".to_owned(),
        "grand ".to_owned(),
        "grand traverse".to_owned(),
        "traverse city".to_owned(),
        "carson city".to_owned(),
        //"abcde".to_owned(),
        //"lmnop".to_owned(),
        //"xyz".to_owned(),
    ]
    .into_iter();
    //let lookup = FuzzyLookup::new(words);
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
            if line_words[1] == "MI" {
                words.insert(line_words[0].to_owned());
            }
        }
    }

    let lookup = FuzzyLookup::new(words.into_iter());
    for _ in 0..10000 {
        i += lookup.lookup("GRADN RAPIDS".to_owned()).unwrap().1;
    }
    println!("{:?}", lookup.lookup("GRAN".to_owned()));
}
