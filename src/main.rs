use ark_ff::{AdditiveGroup, Field};
use ark_test_curves::bls12_381::Fr;
use itertools::Itertools;

use std::{
    collections::HashMap,
    ops::{Add, Mul},
};

fn main() {
    let _ = fr!(5) * EqMLE::new("x", &vec![true, true]) * EqMLE::new("x", &vec![true, false]);

    let f_mle = fr!(22) * EqMLE::new("x", &vec![true, true])
        + fr!(33) * EqMLE::new("x", &vec![true, false])
        + fr!(44) * EqMLE::new("x", &vec![false, true])
        + fr!(55) * EqMLE::new("x", &vec![false, false]);

    assert_eq!(f_mle.clone().evaluate("x", &vec![true, true]).fin(), fr!(22));
    assert_eq!(f_mle.clone().evaluate("x", &vec![true, false]).fin(), fr!(33));
    assert_eq!(f_mle.clone().evaluate("x", &vec![false, true]).fin(), fr!(44));
    assert_eq!(f_mle.clone().evaluate("x", &vec![false, false]).fin(), fr!(55));
}

type ProdTerms = (Fr, Vec<bool>);
#[derive(Clone)]
pub struct EqMLE {
    sum: Vec<(Fr, HashMap<String, ProdTerms>)>,
    coeff: Fr,
}

impl EqMLE {
    pub fn new(variable: &str, booleans: &[bool]) -> Self {
        let mut map = HashMap::new();
        map.insert(
            variable.to_string(),
            (fr!(1), booleans.iter().map(|b| *b).collect()),
        );
        Self {
            sum: vec![(fr!(1), map)],
            coeff: fr!(1),
        }
    }
    pub fn evaluate(self, variable: &str, v: &[bool]) -> Self {
        let mut sum = vec![];

        for sum_term in self.sum {
            let (coeff, mut map) = sum_term;

            let (v_coeff, v_prod_terms) = map.remove(variable).expect("msg");
            // 全て一致していれば、このsum_termは0にはならないので、sumに再び加える。
            if v.iter()
                .zip(v_prod_terms.iter())
                .all(|(v_i, t_i)| v_i == t_i)
            {
                sum.push((coeff * v_coeff, map)) //変数に値を入れて評価して得られた点は係数にまとめる。
            }
        }

        Self {
            sum,
            coeff: self.coeff,
        }
    }

    pub fn fin(self) -> Fr {
        let mut sum = fr!(0);
        for (coeff, map) in self.sum {
            assert!(map.len() == 0);
            sum += coeff
        }

        sum
    }
}

// EqMLE * EqMLE
impl Mul for EqMLE {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut sum = vec![];

        for self_sum_term in &self.sum {
            'outer: for rhs_sum_term in &rhs.sum {
                // mapの共通するvariable同士を掛け合わせる。
                let (coeff_0, map_0) = self_sum_term;
                let (coeff_1, map_1) = rhs_sum_term;

                let mut new_map = HashMap::new();

                // 同じsum_termにある全ての変数に対して。
                for variable in map_0.keys().chain(map_1.keys()).sorted().dedup() {
                    match (map_0.get(variable), map_1.get(variable)) {
                        (None, None) => panic!(),
                        (None, Some(v)) => {new_map.insert(variable.to_string(), v.clone());},
                        (Some(v), None) => {new_map.insert(variable.to_string(), v.clone());},
                        (Some(v0), Some(v1)) => {
                            let (coeff_v0, prod_terms_v0) = v0;
                            let (coeff_v1, prod_terms_v1) = v1;
                            // 掛け合わせたprod_termが0にならないかを調べる。
                            let mut new_prod_terms = vec![];
                            for (b, _b) in prod_terms_v0.into_iter().zip(prod_terms_v1) {
                                if b == _b {
                                    new_prod_terms.push(*b);
                                } else {
                                    // もし、評価した結果が0になるのなら、このsum_termは0になるので、次に移動する。
                                    break 'outer;
                                }
                            }
                            new_map.insert(variable.to_string(), (coeff_v0*coeff_v1, new_prod_terms));
                        },
                    };
                }

                sum.push((coeff_0*coeff_1, new_map))
            }
        }

        EqMLE {
            sum,
            coeff: self.coeff * rhs.coeff,
        }
    }
}

impl Add for EqMLE {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        // 係数をsumの中の係数に掛け合わせて消す
        let sum_0: Vec<_> = self
            .sum
            .into_iter()
            .map(|(coeff, map)| (coeff * self.coeff, map))
            .collect();
        let sum_1: Vec<_> = rhs
            .sum
            .into_iter()
            .map(|(coeff, map)| (coeff * rhs.coeff, map))
            .collect();
        Self {
            sum: sum_0.into_iter().chain(sum_1).collect(),
            coeff: fr!(1),
        }
    }
}

// EqMLE * Fr
impl Mul<Fr> for EqMLE {
    type Output = Self;

    fn mul(mut self, scalar: Fr) -> Self::Output {
        self.coeff *= scalar;
        self
    }
}

// Fr * EqMLE
impl Mul<EqMLE> for Fr {
    type Output = EqMLE;

    fn mul(self, mut rhs: EqMLE) -> Self::Output {
        rhs.coeff *= self;
        rhs
    }
}

fn _all_bit_patterns(n: usize) -> Vec<Vec<Fr>> {
    let total = 1 << n; // 2^n
    let mut result = Vec::with_capacity(total);

    for i in 0..total {
        let mut bits = Vec::with_capacity(n);
        for b in 0..n {
            // b番目のビットが1なら Fr::ONE, 0なら Fr::ZERO
            if (i >> b) & 1 == 1 {
                bits.push(Fr::ONE);
            } else {
                bits.push(Fr::ZERO);
            }
        }
        result.push(bits);
    }

    result
}

pub fn satisfy_poly(x: Fr, coeffs: &[Fr]) -> bool {
    let mut x_n = Fr::ONE;
    let mut sum = Fr::ZERO;
    for &coeff in coeffs {
        sum += coeff * x_n;
        x_n *= x;
    }

    sum == Fr::ZERO
}

#[macro_export]
macro_rules! fr {
    ($val:expr) => {
        ark_test_curves::bls12_381::Fr::from($val)
    };
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn check_satisfy_poly() {
        // x^3 - 9x^2 + 26x - 24 = 0
        let coeffs = vec![fr!(-24), fr!(26), fr!(-9), fr!(1)];

        assert!(satisfy_poly(fr!(2), &coeffs));
        assert!(satisfy_poly(fr!(3), &coeffs));
        assert!(satisfy_poly(fr!(4), &coeffs));

        assert!(!satisfy_poly(fr!(5), &coeffs));
        assert!(!satisfy_poly(fr!(6), &coeffs));
        assert!(!satisfy_poly(fr!(7), &coeffs));
    }
}
