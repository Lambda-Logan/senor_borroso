use std::borrow::Borrow;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::{BuildHasher, Hash, Hasher};
use std::iter::Extend;
use std::ops::{AddAssign, Deref};

use crate::utils::{get_entry, Entry};
use fxhash::FxHasher64;
use std::mem::transmute;

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct Feature(pub u64);

pub trait FeatureFrom<Token> {
    type State: Default;
    fn eat_token(state: &mut Self::State, token: Token);

    fn produce_feature(state: Self::State) -> Self;

    ////////////////////////////////////
    ////////////////////////////////////

    fn eat_token_with_flag(state: &mut Self::State, token: Token, flag: usize) {
        Self::eat_token(state, token);
    }

    fn default_with_flag(flag: usize) -> Self::State {
        Default::default()
    }
}

impl<Token> FeatureFrom<Token> for Vec<Token> {
    type State = Self;
    fn eat_token(state: &mut Self, token: Token) {
        state.push(token);
    }
    fn produce_feature(state: Self) -> Self {
        state
    }
}

impl<Token: Eq + Hash, S: Default + BuildHasher> FeatureFrom<Token> for HashSet<Token, S> {
    type State = Self;
    fn eat_token(state: &mut Self, token: Token) {
        state.insert(token);
    }
    fn produce_feature(state: Self) -> Self {
        state
    }
}

impl<Token, N, S> FeatureFrom<Token> for HashMap<Token, N, S>
where
    Token: Eq + Hash,
    N: Default + AddAssign + From<u8>,
    S: Default + BuildHasher,
{
    type State = Self;
    fn eat_token(state: &mut Self, token: Token) {
        *state.entry(token).or_default() += From::from(1);
    }
    fn produce_feature(state: Self) -> Self {
        state
    }
}
#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct HashedFeature64(u64);

impl<Token: Eq + Hash> FeatureFrom<Token> for HashedFeature64 {
    type State = FxHasher64;

    fn eat_token(state: &mut Self::State, token: Token) {
        &token.hash(state);
    }

    fn produce_feature(state: Self::State) -> Self {
        HashedFeature64(state.finish())
    }
}

impl<TokenPtr, Token, Tokens: IntoIterator<Item = TokenPtr>> FeatureFrom<Tokens> for String
where
    TokenPtr: ToOwned<Owned = Token>,
    String: Extend<Token>,
{
    type State = Self;

    #[inline]
    fn eat_token(state: &mut Self::State, token: Tokens) {
        state.extend(token.into_iter().map(|x| x.to_owned()))
    }
    #[inline]
    fn produce_feature(state: Self::State) -> Self {
        state
    }
    #[inline]
    fn eat_token_with_flag(state: &mut Self::State, token: Tokens, flag: usize) {
        if flag > 0 {
            state.push('_');
        }
        state.extend(token.into_iter().map(|x| x.to_owned()));
    }
}

impl<Token, A, B> FeatureFrom<Token> for (A, B)
where
    Token: Clone,
    A: FeatureFrom<Token>,
    B: FeatureFrom<Token>,
{
    type State = (A::State, B::State);
    fn eat_token(state: &mut Self::State, token: Token) {
        A::eat_token(&mut state.0, token.clone());
        B::eat_token(&mut state.1, token);
    }

    fn produce_feature(state: Self::State) -> Self {
        (A::produce_feature(state.0), B::produce_feature(state.1))
    }

    ////////////////////////////////////
    ////////////////////////////////////

    fn eat_token_with_flag(state: &mut Self::State, token: Token, flag: usize) {
        A::eat_token_with_flag(&mut state.0, token.clone(), flag);
        B::eat_token_with_flag(&mut state.1, token, flag);
    }

    fn default_with_flag(flag: usize) -> Self::State {
        (A::default_with_flag(flag), B::default_with_flag(flag))
    }
}

pub trait Featurizer<TokenGroup: Clone>
where
    Self: Sized,
{
    fn use_tokens_from<Feat, Push, State>(&self, tokengroup: TokenGroup, push_feat: &mut Push)
    where
        Feat: FeatureFrom<TokenGroup, State = State>,
        Push: FnMut(Feat) -> (),
        State: Default;

    fn featurize<Origin, FeatureStep, FeatureGroup>(&self, origin: Origin) -> FeatureGroup
    where
        Origin: HasTokens<TokenGroup = TokenGroup>,
        FeatureStep: FeatureFrom<TokenGroup>,
        FeatureGroup: FeatureFrom<FeatureStep>,
    {
        let mut group: FeatureGroup::State = Default::default();
        {
            let mut push_step = |step: FeatureStep| FeatureGroup::eat_token(&mut group, step);
            //self.run(origin.expose_tokens(), &mut push_step);
            origin.give_tokens_to(self, &mut push_step);
        }
        FeatureGroup::produce_feature(group)
    }

    fn featurize_x2<Origin, FeatureStepA, FeatureGroupA, FeatureStepB, FeatureGroupB>(
        &self,
        origin: Origin,
    ) -> (FeatureGroupA, FeatureGroupB)
    where
        Origin: HasTokens<TokenGroup = TokenGroup>,
        FeatureStepA: FeatureFrom<TokenGroup>,
        FeatureGroupA: FeatureFrom<FeatureStepA>,

        FeatureStepB: FeatureFrom<TokenGroup>,
        FeatureGroupB: FeatureFrom<FeatureStepB>,
    {
        let mut group_A: FeatureGroupA::State = Default::default();
        let mut group_B: FeatureGroupB::State = Default::default();
        {
            let mut push_step = |steps: (FeatureStepA, FeatureStepB)| {
                FeatureGroupA::eat_token(&mut group_A, steps.0);
                FeatureGroupB::eat_token(&mut group_B, steps.1);
            };
            //self.run(origin.expose_tokens(), &mut push_step);
            origin.give_tokens_to(self, &mut push_step);
        }
        (
            FeatureGroupA::produce_feature(group_A),
            FeatureGroupB::produce_feature(group_B),
        )
    }

    fn featurize_x3<
        Origin,
        FeatureStepA,
        FeatureGroupA,
        FeatureStepB,
        FeatureGroupB,
        FeatureStepC,
        FeatureGroupC,
    >(
        &self,
        origin: Origin,
    ) -> (FeatureGroupA, FeatureGroupB, FeatureGroupC)
    where
        Origin: HasTokens<TokenGroup = TokenGroup>,
        FeatureStepA: FeatureFrom<TokenGroup>,
        FeatureGroupA: FeatureFrom<FeatureStepA>,

        FeatureStepB: FeatureFrom<TokenGroup>,
        FeatureGroupB: FeatureFrom<FeatureStepB>,

        FeatureStepC: FeatureFrom<TokenGroup>,
        FeatureGroupC: FeatureFrom<FeatureStepC>,
    {
        let mut group_A: FeatureGroupA::State = Default::default();
        let mut group_B: FeatureGroupB::State = Default::default();
        let mut group_C: FeatureGroupC::State = Default::default();
        {
            let mut push_step = |steps: (FeatureStepA, (FeatureStepB, FeatureStepC))| {
                FeatureGroupA::eat_token(&mut group_A, steps.0);
                FeatureGroupB::eat_token(&mut group_B, steps.1 .0);
                FeatureGroupC::eat_token(&mut group_C, steps.1 .1);
            };
            //self.run(origin.expose_tokens(), &mut push_step);
            origin.give_tokens_to(self, &mut push_step);
        }
        (
            FeatureGroupA::produce_feature(group_A),
            FeatureGroupB::produce_feature(group_B),
            FeatureGroupC::produce_feature(group_C),
        )
    }
}

pub trait HasTokens {
    type TokenGroup: Clone;
    fn give_tokens_to<Ftzr, Push, Feat>(&self, ftzr: &Ftzr, push_feat: &mut Push)
    where
        Ftzr: Featurizer<Self::TokenGroup>,
        Push: FnMut(Feat) -> (),
        Feat: FeatureFrom<Self::TokenGroup>;
}

impl<T> HasTokens for &[T] {
    type TokenGroup = Self;
    fn give_tokens_to<Ftzr, Push, Feat>(&self, ftzr: &Ftzr, push_feat: &mut Push)
    where
        Ftzr: Featurizer<Self::TokenGroup>,
        Feat: FeatureFrom<Self::TokenGroup>,
        Push: FnMut(Feat) -> (),
    {
        ftzr.use_tokens_from(&self, push_feat);
    }
}

impl<'a, T> HasTokens for &'a Vec<T> {
    type TokenGroup = &'a [T];
    fn give_tokens_to<Ftzr, Push, Feat>(&self, ftzr: &Ftzr, push_feat: &mut Push)
    where
        Ftzr: Featurizer<Self::TokenGroup>,
        Feat: FeatureFrom<Self::TokenGroup>,
        Push: FnMut(Feat) -> (),
    {
        ftzr.use_tokens_from(&self, push_feat);
    }
}

impl<'a> HasTokens for &'a str {
    type TokenGroup = &'a [char];
    fn give_tokens_to<Ftzr, Push, Feat>(&self, ftzr: &Ftzr, push_feat: &mut Push)
    where
        Ftzr: Featurizer<Self::TokenGroup>,
        Feat: FeatureFrom<Self::TokenGroup>,
        Push: FnMut(Feat) -> (),
    {
        //ftzr.r
        let v: Vec<_> = Iterator::collect(self.chars());
        let va = unsafe { transmute::<&Vec<char>, &'a Vec<char>>(&v) };
        ftzr.use_tokens_from(va, push_feat);
    }
}

impl<'a> HasTokens for &'a String {
    type TokenGroup = <&'a str as HasTokens>::TokenGroup;
    fn give_tokens_to<Ftzr, Push, Feat>(&self, ftzr: &Ftzr, push_feat: &mut Push)
    where
        Ftzr: Featurizer<Self::TokenGroup>,
        Feat: FeatureFrom<Self::TokenGroup>,
        Push: FnMut(Feat) -> (),
    {
        self.as_str().give_tokens_to(ftzr, push_feat);
    }
}

pub trait CanGram {
    #[inline]
    fn run<T: Sized + Hash + Debug, F: FnMut(Feature) -> ()>(&self, s: &[T], push_feat: &mut F);
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct SkipScheme {
    pub(crate) group_a: (usize, usize),
    pub(crate) gap: (usize, usize),
    pub(crate) group_b: (usize, usize),
}

impl<'a, T> Featurizer<&'a [T]> for SkipScheme {
    #[inline]
    fn use_tokens_from<Feat, Push, State>(&self, tokengroup: &'a [T], push_feat: &mut Push)
    where
        Feat: FeatureFrom<&'a [T], State = State>,
        Push: FnMut(Feat) -> (),
        State: Default,
    {
        let s = tokengroup;
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
                            //let mut hasher: FxHasher64 = Default::default();
                            let mut eater: State = Default::default();

                            if group_a.len() != 0 {
                                //group_a.hash(&mut hasher);
                                Feat::eat_token_with_flag(&mut eater, group_a, 0);
                            };
                            if group_b.len() != 0 {
                                //group_b.hash(&mut hasher);
                                Feat::eat_token_with_flag(&mut eater, group_b, 1);
                            };
                            //updt(Feature(hasher.finish()));
                            push_feat(Feat::produce_feature(eater));
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
                        //let mut hasher: FxHasher64 = Default::default();
                        let mut eater: State = Default::default();
                        //&s[x..y].hash(&mut hasher);
                        Feat::eat_token(&mut eater, &s[x..y]);
                        //updt(Feature(hasher.finish()));
                        push_feat(Feat::produce_feature(eater));
                    };
                }
            }
        }
    }
}

impl CanGram for SkipScheme {
    #[inline]
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
    #[inline]
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
    #[inline]
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

impl<'a, T, Ftzr: Featurizer<&'a [T]>> Featurizer<&'a [T]> for BookEndsFtzr<Ftzr> {
    #[inline]
    fn use_tokens_from<Feat, Push, State>(&self, tokengroup: &'a [T], push_feat: &mut Push)
    where
        Feat: FeatureFrom<&'a [T], State = State>,
        Push: FnMut(Feat) -> (),
        State: Default,
    {
        //let mut pf = |n: Feature| push_feat(Feature(BookEnds::Head(n.0).uniq()));
        {
            //let mut eater: State = Default::default();
            if tokengroup.len() >= self.head {
                //println!("head {:?}", &s[..self.head]);
                //Feat::eat_token_with_flag(&mut eater, &tokengroup[..self.head], 0);
                self.ftzr
                    .use_tokens_from(&tokengroup[..self.head], push_feat);
                //self.ftzr.run(&s[..self.head], &mut pf);
            }
            //push_feat(Feat::produce_feature(eater));
        }

        //let mut pf = |n: Feature| push_feat(Feature(BookEnds::Toe(n.0).uniq()));
        {
            //let mut eater: State = Default::default();
            if tokengroup.len() >= self.toe {
                //println!("toe {:?}", &s[(s.len() - self.toe)..s.len()]);
                //self.ftzr.run(&s[(s.len() - self.toe)..s.len()], &mut pf);
                //Feat::eat_token_with_flag(&mut eater,&tokengroup[(tokengroup.len() - self.toe)..tokengroup.len()],1,);
                self.ftzr.use_tokens_from(
                    &tokengroup[(tokengroup.len() - self.toe)..tokengroup.len()],
                    push_feat,
                );
            }
            //push_feat(Feat::produce_feature(eater));
        }
    }
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct EmptyFtzr;

impl CanGram for EmptyFtzr {
    #[inline]
    fn run<T: Sized + Hash + Debug, F: FnMut(Feature) -> ()>(&self, s: &[T], push_feat: &mut F) {}
}

impl<T: Clone> Featurizer<T> for EmptyFtzr {
    #[inline]
    fn use_tokens_from<Feat, Push, State>(&self, tokengroup: T, push_feat: &mut Push)
    where
        Feat: FeatureFrom<T, State = State>,
        Push: FnMut(Feat) -> (),
        State: Default,
    {
    }
}

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq, Debug)]
pub struct MultiFtzr<A, B> {
    pub a: A,
    pub b: B,
}

impl<A: CanGram, B: CanGram> CanGram for MultiFtzr<A, B> {
    #[inline]
    fn run<T: Sized + Hash + Debug, F: FnMut(Feature) -> ()>(&self, s: &[T], push_feat: &mut F) {
        self.a.run(&s, push_feat);
        self.b.run(&s, push_feat);
    }
}

impl<T, A, B> Featurizer<T> for MultiFtzr<A, B>
where
    T: Clone,
    A: Featurizer<T>,
    B: Featurizer<T>,
{
    #[inline]
    fn use_tokens_from<Feat, Push, State>(&self, tokengroup: T, push_feat: &mut Push)
    where
        Feat: FeatureFrom<T, State = State>,
        Push: FnMut(Feat) -> (),
        State: Default,
    {
        self.a.use_tokens_from(tokengroup.clone(), push_feat);
        self.b.use_tokens_from(tokengroup, push_feat);
    }
}
pub struct Mapped<Ftzr: CanGram> {
    pub(crate) table: Vec<Entry<Feature, Feature>>,
    pub(crate) keep_unseen: bool,
    pub ftzr: Ftzr,
}

impl<Ftzr: CanGram> CanGram for Mapped<Ftzr> {
    #[inline]
    fn run<T: Sized + Hash + Debug, F: FnMut(Feature) -> ()>(&self, s: &[T], push_feat: &mut F) {
        let mut push_mapped_feat = |feat_a: Feature| {
            let feat_b_opt = get_entry(&self.table, &feat_a).map(|e| e.1.entry);
            match feat_b_opt {
                None => {
                    if self.keep_unseen {
                        push_feat(feat_a)
                    }
                }
                Some(feat_b) => push_feat(feat_b),
            }
        };
        self.ftzr.run(&s, &mut push_mapped_feat);
    }
}

impl<Ftzr: CanGram> Mapped<Ftzr> {
    pub fn from_filtermap<Fm, Feats>(f: Fm, feats: Feats, ftzr: Ftzr) -> Self
    where
        Fm: Fn(Feature) -> Option<Feature>,
        Feats: Iterator<Item = Feature>,
    {
        let mut featset: HashSet<Feature> = Default::default();
        let mut table: Vec<Entry<Feature, Feature>> = Default::default();
        for feat_a in feats {
            if !featset.contains(&feat_a) {
                match f(feat_a) {
                    Some(feat_b) => {
                        table.push(Entry {
                            id: feat_a,
                            entry: feat_b,
                        });
                        featset.insert(feat_a);
                    }
                    _ => (),
                }
            }
        }

        table.sort();

        Mapped {
            table: table,
            ftzr: ftzr,
            keep_unseen: false,
        }
    }
}
