use ark_ff::{AdditiveGroup, Field};
use ark_test_curves::bls12_381::Fr;

use std::{num::NonZero, ops::Add, ops::Mul};

fn main() {
    let aa = fr!(5) * EqMLE::new("x", &vec![true, true]) * EqMLE::new("x", &vec![true, false]);

    let aaa = fr!(2) * EqMLE::new("x", &vec![true, true])
        + fr!(3) * EqMLE::new("x", &vec![true, false])
        + fr!(4) * EqMLE::new("x", &vec![false, true])
        + fr!(5) * EqMLE::new("x", &vec![false, false]);
}

pub enum EqTerm {
    Zero,
    Bool(bool),
}

pub struct EqMLE {
    // name: String,
    sum: Vec<(Fr, Vec<bool>)>,
    coeff: Fr,
}

impl EqMLE {
    pub fn new(_name: &str, booleans: &[bool]) -> Self {
        Self {
            // name: name.to_string(),
            sum: vec![(fr!(1), booleans.to_vec())],
            coeff: fr!(1),
        }
    }
}

// EqMLE * EqMLE
impl Mul for EqMLE {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut sum = vec![];

        for (coeff_0, eq_prod_0) in self.sum {
            'outer: for (coeff_1, eq_prod_1) in &rhs.sum {
                let mut terms = vec![];
                for (b, _b) in eq_prod_0.iter().zip(eq_prod_1) {
                    if b == _b {
                        terms.push(*b)
                    } else {
                        // もし二つのbooleanが異なれば、そのeq~のproductは全て0になるので、sumには加えない。
                        break 'outer;
                    };
                }
                sum.push((coeff_0 * coeff_1, terms));
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
            .map(|(coeff, prod_terms)| (coeff * self.coeff, prod_terms))
            .collect();
        let sum_1: Vec<_> = rhs
            .sum
            .into_iter()
            .map(|(coeff, prod_terms)| (coeff * rhs.coeff, prod_terms))
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

fn all_bit_patterns(n: usize) -> Vec<Vec<Fr>> {
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
