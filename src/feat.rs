use crate::dualiter::*;
use fxhash::{FxHasher32, FxHasher64};
//use probabilistic_collections::similarity::SimHash;
//use probabilistic_collections::SipHasherBuilder;
use space::{Hamming, MetricPoint};
use std::cmp;
use std::fmt;
use std::fmt::Debug;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::marker::PhantomData;
use std::mem::transmute;
use std::str;
use unidecode::unidecode;

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
//type FnvHashMap<K, V> = HashMap<K, V, FnvBuildHasher>;

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct FuzzyEntry<T, Id = String> {
    pub id: Id,
    pub feats: Vec<T>,
    simhash: u128,
}

fn hash_feature<T: Hash>(t: &T) -> u64 {
    let mut s = FnvHasher::default();
    t.hash(&mut s);
    s.finish()
}

/// Calculate `u64` simhash from stream of `&str` words
pub fn simhash_stream<W, T: Hash>(words: W) -> u64
where
    W: Iterator<Item = T>,
{
    let mut v = [0i32; 64];
    let mut simhash: u64 = 0;

    for feature in words {
        let feature_hash: u64 = hash_feature(&feature);

        for i in 0..64 {
            let bit = (feature_hash >> i) & 1;
            if bit == 1 {
                v[i] = v[i].saturating_add(1);
            } else {
                v[i] = v[i].saturating_sub(1);
            }
        }
    }

    for q in 0..64 {
        if v[q] > 0 {
            simhash |= 1 << q;
        }
    }
    simhash
}

pub fn simhash_stream_u128<W, T: Hash>(words: W) -> u128
where
    W: Iterator<Item = T>,
{
    let mut v = [0i32; 128];
    let mut simhash: u128 = 0;

    for feature in words {
        let feature_hash: u64 = hash_feature(&feature);

        for i in 0..128 {
            let bit = (feature_hash >> i) & 1;
            if bit == 1 {
                v[i] = v[i].saturating_add(1);
            } else {
                v[i] = v[i].saturating_sub(1);
            }
        }
    }

    for q in 0..64 {
        if v[q] > 0 {
            simhash |= 1 << q;
        }
    }
    simhash
}

pub type FeatEntry<F: Featurizer> = FuzzyEntry<F::Feat, F::Id>;

pub trait Featurizer {
    type Origin: Ord + Hash + Clone + Debug;
    type Feat: Ord + Hash + Clone + Debug;
    type Id: Ord + Hash + Clone + Debug;

    fn push_features<F: FnMut(Self::Feat) -> ()>(
        &self,
        s: Self::Origin,
        push_feat: &mut F,
    ) -> Self::Id;

    fn new(&self, s: Self::Origin) -> FuzzyEntry<Self::Feat, Self::Id> {
        let mut vbow: Vec<Self::Feat> = Vec::with_capacity(8);
        let id = {
            let mut push_feat = |f: Self::Feat| vbow.push(f);
            self.push_features(s, &mut push_feat)
        };

        vbow.sort();

        let sim_a = 0;
        //SimHash::with_hasher(SipHasherBuilder::from_seed(0, 0)).get_sim_hash(vbow.iter());
        //SimHash::with_hasher(SipHasherBuilder::from_seed(0, 0)).get_sim_hash(vbow.iter());
        //let sim_b = SimHash::with_hasher(SipHasherBuilder::from_seed(123456789, 123456789))
        //    .get_sim_hash(vbow.iter());
        let sim_b = 0; //simhash_stream(vbow.iter());
                       //let sh = simhash_stream_u128(vbow.iter());
        let r = FuzzyEntry {
            id: id,
            feats: vbow,
            simhash: 0, //unsafe { transmute((sim_a, sim_b)) },
        };
        r
    }
}
impl<T: Ord, Id> FuzzyEntry<T, Id> {
    #[inline(always)]
    pub fn sim(&self, other: &Self) -> f64 {
        //return 1.0 / self.ham_dist(other) as f64;
        let mut n = 0;
        let mut d = 0;
        for x in DualIter::new(self.feats.iter(), other.feats.iter()) {
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
        (n as f64 / d as f64)
    }
    pub fn ham_dist(&self, other: &Self) -> u64 {
        Hamming(self.simhash).distance(&Hamming(other.simhash))
    }
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq)]
pub struct AsciiFeat(u8, u8, u8, u8);

impl fmt::Debug for AsciiFeat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a_val = &[self.0];
        let b_val = &[self.1];
        let c_val = &[self.2];
        let d_val = &[self.3];
        let a = str::from_utf8(a_val).unwrap();
        let b = str::from_utf8(b_val).unwrap();
        let c = str::from_utf8(c_val).unwrap();
        let d = str::from_utf8(d_val).unwrap();
        f.debug_tuple("AsciiFeat")
            .field(&a)
            .field(&b)
            .field(&c)
            .field(&d)
            .finish()
    }
}

#[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct DefaultAscii;

impl Featurizer for DefaultAscii {
    type Origin = String;
    type Feat = AsciiFeat;
    type Id = String;

    fn push_features<F>(&self, origin: String, push_feat: &mut F) -> String
    where
        F: FnMut(AsciiFeat) -> (),
    {
        let default: u8 = 35;
        let s = origin.as_bytes();
        let l = s.len();
        if l > 1 {
            push_feat(AsciiFeat(36, s[0], s[1], default));
            push_feat(AsciiFeat(default, s[l - 2], s[l - 1], 36));
        }
        if l > 2 {
            push_feat(AsciiFeat(36, s[1], s[2], default));
            push_feat(AsciiFeat(default, s[l - 3], s[l - 2], 36));
        }
        for c in 1..l {
            let b = c - 1;
            let _a = cmp::max(0, (b as isize) - 3) as usize;
            for a in _a..b {
                if a > 0 {
                    let pre_a = a - 1;
                    push_feat(AsciiFeat(s[pre_a], s[a], s[b], s[c]));
                } else {
                    push_feat(AsciiFeat(s[a], s[b], s[c], default));
                }
            }
        }
        origin
    }
}

pub trait CanGram {
    fn run<F: FnMut(u64) -> (), T: Sized + Hash + fmt::Debug>(&self, s: &[T], push_feat: &mut F);
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct SkipScheme {
    pub group_a: (usize, usize),
    pub gap: (usize, usize),
    pub group_b: (usize, usize),
}

impl CanGram for SkipScheme {
    fn run<F, T: Sized + Hash + fmt::Debug>(&self, s: &[T], updt: &mut F)
    where
        F: FnMut(u64) -> (),
    {
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
                            updt(hasher.finish());
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
                        let mut hasher: FnvHasher = Default::default();
                        &s[x..y].hash(&mut hasher);
                        updt(hasher.finish());
                    };
                }
            }
        }
    }
}

impl<Cg: CanGram> CanGram for [Cg] {
    fn run<F: FnMut(u64) -> (), T: Sized + Hash + fmt::Debug>(&self, s: &[T], push_feat: &mut F) {
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
    fn run<F: FnMut(u64) -> (), T: Sized + Hash + fmt::Debug>(&self, s: &[T], push_feat: &mut F) {
        {
            let mut pf = |n: u64| push_feat(BookEnds::Head(n).uniq());
            if s.len() >= self.head {
                //println!("head {:?}", &s[..self.head]);
                self.ftzr.run(&s[..self.head], &mut pf);
            }
        };
        {
            let mut pf = |n: u64| push_feat(BookEnds::Toe(n).uniq());
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
    fn run<F: FnMut(u64) -> (), T: Sized + Hash + fmt::Debug>(&self, s: &[T], push_feat: &mut F) {}
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct MultiFtzr<A, B> {
    pub a: A,
    pub b: B,
}

impl<A: CanGram, B: CanGram> CanGram for MultiFtzr<A, B> {
    fn run<F: FnMut(u64) -> (), T: Sized + Hash + fmt::Debug>(&self, s: &[T], push_feat: &mut F) {
        self.a.run(&s, push_feat);
        self.b.run(&s, push_feat);
    }
}

macro_rules! featurizers {
    () => {
        (EmptyFtzr)
    };
    ($a:expr $(, $tail:expr)*) => {{
        MultiFtzr {
            a: $a,
            b: featurizers!($($tail), *),
        }
    }};
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct Token(u64);

#[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct DefaultUnicode;

impl Featurizer for DefaultUnicode {
    type Origin = String;
    type Feat = Token;
    type Id = String;

    fn push_features<F>(&self, origin: String, push_feat: &mut F) -> String
    where
        F: FnMut(Token) -> (),
    {
        let mut _updt = |u| push_feat(Token(u));
        //let mut xs: Vec<u8> = Vec::with_capacity(origin.len() + 2);
        //xs.push(0);
        let xs = unidecode(&origin).as_bytes().to_owned();
        //xs.push('#');
        //xs.extend(origin.chars());
        //xs.push('$');
        //xs.push(255);
        let ss = SkipScheme {
            group_a: (2, 2), //(2, 3),
            gap: (0, 3),
            group_b: (2, 2),
        };
        //ss.run(&xs, &mut _updt);
        //book_ends((4, 4), n_gram(2))
        featurizers![
            SkipScheme {
                group_a: (2, 2), //(2, 3),
                gap: (0, 3),
                group_b: (2, 2),
            },
            book_ends((4, 4), n_gram(2))
        ]
        .run(&xs, &mut _updt);
        origin
    }
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct AnonFtzr<
    G_Id: Fn(Origin) -> Id,
    G_T: Fn(&Origin) -> Vec<Tok>,
    U_T: CanGram,
    Origin,
    Id,
    Tok,
> {
    pub get_id: G_Id,
    pub get_tokens: G_T,
    pub use_tokens: U_T,
    pub meta: PhantomData<(Origin, Id, Tok)>,
}

impl<G_Id, G_T, U_T, Origin, Id, Tok> Featurizer for AnonFtzr<G_Id, G_T, U_T, Origin, Id, Tok>
where
    //Feat: Hash + Debug + Ord + Clone,
    Origin: Hash + Debug + Ord + Clone,
    Id: Hash + Debug + Ord + Clone,
    G_Id: Fn(Origin) -> Id,
    G_T: Fn(&Origin) -> Vec<Tok>,
    U_T: CanGram,
    Tok: Sized + Hash + Debug,
{
    type Origin = Origin;
    type Feat = Token;
    type Id = Id;
    fn push_features<F: FnMut(Self::Feat) -> ()>(
        &self,
        s: Self::Origin,
        push_feat: &mut F,
    ) -> Self::Id {
        let mut _updt = |u| push_feat(Token(u));
        let doc = &(self.get_tokens)(&s);
        self.use_tokens.run(doc, &mut _updt);
        (self.get_id)(s)
    }
}

#[derive(Hash, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct Doc<Id, Tok> {
    pub id: Id,
    pub body: Vec<Tok>,
}

#[derive(Hash, Clone, PartialEq, Ord, PartialOrd, Eq, Debug, Copy)]
pub struct DocFtzr<Cg: CanGram, Id, Tok> {
    use_tokens: Cg,
    meta: PhantomData<(Id, Tok)>,
}

impl<Cg: CanGram, Id, Tok> DocFtzr<Cg, Id, Tok> {
    pub fn new(cg: Cg) -> Self {
        DocFtzr {
            use_tokens: cg,
            meta: Default::default(),
        }
    }
}

impl<Cg: CanGram, Id, Tok> Featurizer for DocFtzr<Cg, Id, Tok>
where
    Tok: Ord + Hash + Clone + Debug,
    Id: Ord + Hash + Clone + Debug,
{
    type Origin = Doc<Id, Tok>;
    type Feat = u64;
    type Id = Id;

    fn push_features<F>(&self, origin: Doc<Id, Tok>, push_feat: &mut F) -> Self::Id
    where
        F: FnMut(u64) -> (),
    {
        self.use_tokens.run(&origin.body, push_feat);
        origin.id
    }
}
