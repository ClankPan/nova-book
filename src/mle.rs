use ark_ff::{AdditiveGroup, Field};
use ark_test_curves::bls12_381::Fr;
use itertools::Itertools;

use core::str;
use std::{
    collections::HashMap,
    env::var,
    ops::{Add, Mul},
};

use crate::{bvec, fr};

#[derive(PartialEq, Debug)]
pub struct Term(Fr, Eq);
pub struct Poly {
    coeff: Fr,
    sum: Vec<(Fr, Eq)>,
}

pub fn vector_mle(vector: &[Fr]) {
    let mle = vector[0] * Eq::x(&bvec![0,0]) + vector[1] * Eq::x(&bvec![0,1]);
}

impl Poly {
    pub fn  eval_x(self, values: &Vec<Option<Fr>>) -> Self {

        let mut sum = vec![];

        for (mut coeff, eq) in self.sum {
            let (x, y)  = eq.inner;
            match x {
                Some(x) => {
                    assert!(values.len() == x.len());

                    let mut new_x = vec![];
                    for (x_i, v_i) in x.iter().zip(values) {
                        if let Some(v_i) = v_i {
                            let x_i = Fr::from(*x_i);
                            coeff *= (Fr::ONE - x_i) * (Fr::ONE - v_i) + x_i * v_i;
                        } else {
                            new_x.push(*x_i)
                        }
                    }

                    sum.push((coeff, Eq{inner: (Some(new_x), y)}))
                },
                None => {
                    sum.push((coeff, Eq{inner: (None, y)}))
                },
            }

        }
        Self {
            coeff: self.coeff,
            sum,
        }
    }
}

// Poly * Fr -> Poly
impl Mul<Fr> for Poly {
    type Output = Self;

    fn mul(mut self, scalar: Fr) -> Self::Output {
        self.coeff *= scalar;
        self
    }
}

// Fr * Poly -> Poly
impl Mul<Poly> for Fr {
    type Output = Poly;

    fn mul(self, mut rhs: Poly) -> Self::Output {
        rhs.coeff *= self;
        rhs
    }
}

// Poly * Poly -> Poly
impl Mul<Poly> for Poly {
    type Output = Self;

    fn mul(self, rhs: Poly) -> Self::Output {
        let coeff = self.coeff * rhs.coeff;
        let mut sum = vec![];
        for (coeff_a, term_a) in self.sum {
            'inner: for (coeff_b, term_b) in &rhs.sum {
                let coeff_ab = coeff_a * coeff_b;
                if coeff_ab == Fr::ZERO { // coeffが0の時はsumから除外する。
                    continue 'inner;
                }
                let term_ab = term_a.clone() * term_b.clone();
                sum.push((coeff_ab, term_ab));
            }
        }
        Self {
            coeff,
            sum
        }
    }
}

// Poly + Poly -> Poly
impl Add<Poly> for Poly {
    type Output = Self;

    fn add(self, rhs: Poly) -> Self::Output {
        let mut sum = vec![];

        for (coeff, term) in self.sum {
            sum.push((coeff*self.coeff, term))
        }

        for (coeff, term) in rhs.sum {
            sum.push((coeff*rhs.coeff, term))
        }

        Self {
            coeff: Fr::ONE,
            sum
        }
    }
}



#[derive(PartialEq, Debug, Clone)]
pub struct Eq {
    inner: (Option<Vec<bool>>, Option<Vec<bool>>),
}

impl Eq {
    pub fn x(inner: &[bool]) -> Self {
        Self {
            inner: (Some(inner.to_vec()), None),
        }
    }
    pub fn y(inner: &[bool]) -> Self {
        Self {
            inner: (None, Some(inner.to_vec())),
        }
    }
}

// Eq * Eq -> Eq
impl Mul<Eq> for Eq {
    type Output = Self;

    fn mul(self, rhs: Eq) -> Self::Output {
        let inner = match (self.inner, rhs.inner) {
            ((Some(x), None), (None, Some(y))) | ((None, Some(x)), (Some(y), None)) => {
                (Some(x), Some(y))
            }
            _ => todo!(),
        };

        Self { inner }
    }
}

// Eq * Fr -> Poly
impl Mul<Fr> for Eq {
    type Output = Poly;

    fn mul(self, coeff: Fr) -> Self::Output {
        Poly {
            coeff,
            sum: vec![(Fr::ONE, self)]
        }
    }
}

// Fr * Eq -> Poly
impl Mul<Eq> for Fr {
    type Output = Poly;

    fn mul(self, rhs: Eq) -> Self::Output {
        // Term(self, rhs)
        Poly {
            coeff: self,
            sum: vec![(Fr::ONE, rhs)]
        }
    }
}


#[cfg(test)]
pub mod tests {
    use crate::{bvec, fr};

    use super::*;

    // #[test]
    // fn check_mul() {
    //     assert_eq!(
    //         fr!(2) * Eq::x(&bvec![0, 1]),
    //         Term(fr!(2), Eq::x(&bvec![0, 1]))
    //     );
    // }
}
