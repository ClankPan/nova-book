use ark_ff::{AdditiveGroup, Field};
use ark_test_curves::bls12_381::Fr;

use std::{num::NonZero, ops::Mul};

fn main() {
    let aa = fr!(5) * EqMLE::new(vec![fr!(0), fr!(1)]) * EqMLE::new(vec![fr!(0), fr!(1)]);
}

pub enum EqTerm {
    Zero,
    Bool(bool)
}

pub struct EqMLE {
    // name: String,
    sum: Vec<(Fr, Vec<EqTerm>)>,
    coeff: Fr
}

impl EqMLE {
    pub fn new(_name: &str, booleans: &[bool]) -> Self {
        Self {
            // name: name.to_string(),
            sum: vec![(fr!(1), booleans.iter().map(|b|EqTerm::Bool(*b)).collect())],
            coeff: fr!(1),
        }
    }
}

// EqMLE * EqMLE
impl Mul for EqMLE {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {

        let mut sum = vec![];
        
        for (coeff_x, prod_x) in self.sum {
            for (coeff_y, prod_y) in &rhs.sum {
                for t in prod_x.iter().zip(prod_y) {
                    match t {
                        (EqTerm::Zero, EqTerm::Zero) => EqTerm::Zero,
                        (EqTerm::Zero, EqTerm::Bool(_)) => EqTerm::Zero,
                        (EqTerm::Bool(_), EqTerm::Zero) => EqTerm::Zero,
                        (EqTerm::Bool(b), EqTerm::Bool(_b)) => {
                            if b == _b {
                                EqTerm::Bool(*b)
                            } else {
                                EqTerm::Zero
                            }
                        },
                    };
                }
                // let c = prod_x.iter().zip(prod_y).map(|t| {
                //     match t {
                //         (EqTerm::Zero, EqTerm::Zero) => EqTerm::Zero,
                //         (EqTerm::Zero, EqTerm::Bool(_)) => EqTerm::Zero,
                //         (EqTerm::Bool(_), EqTerm::Zero) => EqTerm::Zero,
                //         (EqTerm::Bool(b), EqTerm::Bool(_b)) => {
                //             if b == _b {
                //                 EqTerm::Bool(*b)
                //             } else {
                //                 EqTerm::Zero
                //             }
                //         },
                //     }
                // }).collect();
                // sum.push((coeff_x*coeff_y, c))
            }
        }
        EqMLE {
            sum,
            coeff: self.coeff * rhs.coeff,
            // name: ,
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