use ark_ff::{AdditiveGroup, Field};
use ark_test_curves::bls12_381::Fr;
use itertools::Itertools;

use std::ops::{Add, AddAssign, Mul};

use crate::{all_bit_patterns, bvec, fr};

#[derive(PartialEq, Debug, Clone)]
pub struct Term(Fr, Eq);

#[derive(PartialEq, Debug, Clone)]
pub struct Poly {
    coeff: Fr,
    sum: Vec<(Fr, Eq)>,
}

pub enum Var {
    X,
    Y,
}

pub fn vector_mle(vector: &[Fr], var: Var) -> Poly {
    let eq = match var {
        Var::X => Eq::x,
        Var::Y => Eq::y,
    };

    let mut mle = Poly::new();
    for (i, pattern) in all_bit_patterns(vector.len()).into_iter().enumerate() {
        mle += vector[i] * eq(&pattern);
    }

    mle
}

pub fn matrix_mle(matrix: &Vec<Vec<Fr>>) -> Poly {
    let m = matrix.len();
    let n = matrix[0].len();

    let mut mle = Poly::new();
    for (i, x_pattern) in all_bit_patterns(m).into_iter().enumerate() {
        for (j, y_pattern) in all_bit_patterns(n).into_iter().enumerate() {
            assert!(matrix[i].len() == n);
            mle += matrix[i][j] * Eq::x(&x_pattern) * Eq::y(&y_pattern);
        }
    }

    mle
}


impl Poly {
    pub fn new() -> Self {
        Self {
            coeff: Fr::ONE,
            sum: vec![],
        }
    }
    pub fn fin(self) -> Fr {
        let mut sum = Fr::ZERO;
        for (coeff, eq) in self.sum {
            assert!(eq.inner.0.is_none() & eq.inner.1.is_none()); // eqには係数しかないことを確認
            sum += coeff;
        }

        sum
    }

    fn eval(self, values: &Vec<Option<Fr>>, is_x: bool) -> Self {
        let mut sum = vec![];

        for (mut coeff, eq) in self.sum {
            let (x, y) = eq.inner;

            let (a, b) = if is_x { (x, y) } else { (y, x) };

            let a = match a {
                Some(a) => {
                    assert!(values.len() == a.len());

                    let mut new_a = vec![];

                    for (a_i, v_i) in a.iter().zip(values) {
                        if let Some(v_i) = v_i {
                            let a_i = Fr::from(*a_i);
                            coeff *= (Fr::ONE - a_i) * (Fr::ONE - v_i) + a_i * v_i;
                        } else {
                            new_a.push(*a_i)
                        }
                    }
                    if new_a.len() == 0 {
                        None
                    } else {
                        Some(new_a)
                    }
                }
                None => None,
            };

            let (x, y) = if is_x { (a, b) } else { (b, a) };
            sum.push((coeff, Eq { inner: (x, y) }))
        }
        Self {
            coeff: self.coeff,
            sum,
        }
    }

    pub fn eval_x(self, values: &Vec<Option<Fr>>) -> Self {
        self.eval(values, true)
    }

    pub fn eval_y(self, values: &Vec<Option<Fr>>) -> Self {
        self.eval(values, false)
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

// Poly * Eq -> Poly
impl Mul<Eq> for Poly {
    type Output = Self;

    fn mul(mut self, eq: Eq) -> Self::Output {
        let mut sum = vec![];
        for (coeff, term) in self.sum {
            sum.push((coeff, term * eq.clone()))
        }
        self.sum = sum;
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
                if coeff_ab == Fr::ZERO {
                    // coeffが0の時はsumから除外する。
                    continue 'inner;
                }
                let term_ab = term_a.clone() * term_b.clone();
                sum.push((coeff_ab, term_ab));
            }
        }
        Self { coeff, sum }
    }
}

// Poly + Poly -> Poly
impl Add<Poly> for Poly {
    type Output = Self;

    fn add(self, rhs: Poly) -> Self::Output {
        let mut sum = vec![];

        for (coeff, term) in self.sum {
            sum.push((coeff * self.coeff, term))
        }

        for (coeff, term) in rhs.sum {
            sum.push((coeff * rhs.coeff, term))
        }

        Self {
            coeff: Fr::ONE,
            sum,
        }
    }
}

impl AddAssign for Poly {
    fn add_assign(&mut self, rhs: Self) {
        for (coeff, _) in &mut self.sum {
            *coeff *= self.coeff;
        }

        for (coeff, term) in rhs.sum {
            self.sum.push((coeff * rhs.coeff, term))
        }

        self.coeff = Fr::ONE;
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
            sum: vec![(Fr::ONE, self)],
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
            sum: vec![(Fr::ONE, rhs)],
        }
    }
}

#[cfg(test)]
pub mod tests {
    use crate::{bvec, fr, frmatrix, frvec};

    use super::*;

    #[test]
    fn check_mle_vector() {
        let vector = frvec![11, 22, 33, 44];
        let mle = vector_mle(&vector, Var::X);

        let x_variables = bvec![0, 0].into_iter().map(|b| Some(Fr::from(b))).collect();
        assert_eq!(mle.clone().eval_x(&x_variables).fin(), fr!(11));

        let x_variables = bvec![0, 1].into_iter().map(|b| Some(Fr::from(b))).collect();
        assert_eq!(mle.clone().eval_x(&x_variables).fin(), fr!(22));

        let x_variables = bvec![1, 0].into_iter().map(|b| Some(Fr::from(b))).collect();
        assert_eq!(mle.clone().eval_x(&x_variables).fin(), fr!(33));

        let x_variables = bvec![1, 1].into_iter().map(|b| Some(Fr::from(b))).collect();
        assert_eq!(mle.clone().eval_x(&x_variables).fin(), fr!(44));
    }

    #[test]
    fn check_mle_matrix() {
        let matrix = frmatrix![
            [11, 12, 13, 14],
            [21, 22, 23, 24],
            [31, 32, 33, 34],
            [41, 42, 43, 44]
        ];
        let mle = matrix_mle(&matrix);

        let v_00 = bvec![0, 0].into_iter().map(|b| Some(Fr::from(b))).collect();
        let v_01 = bvec![0, 1].into_iter().map(|b| Some(Fr::from(b))).collect();
        let v_10 = bvec![1, 0].into_iter().map(|b| Some(Fr::from(b))).collect();
        let v_11 = bvec![1, 1].into_iter().map(|b| Some(Fr::from(b))).collect();

        assert_eq!(mle.clone().eval_x(&v_00).eval_y(&v_00).fin(), fr!(11));
        assert_eq!(mle.clone().eval_x(&v_00).eval_y(&v_01).fin(), fr!(12));
        assert_eq!(mle.clone().eval_x(&v_00).eval_y(&v_10).fin(), fr!(13));
        assert_eq!(mle.clone().eval_x(&v_00).eval_y(&v_11).fin(), fr!(14));

        assert_eq!(mle.clone().eval_x(&v_01).eval_y(&v_00).fin(), fr!(21));
        assert_eq!(mle.clone().eval_x(&v_01).eval_y(&v_01).fin(), fr!(22));
        assert_eq!(mle.clone().eval_x(&v_01).eval_y(&v_10).fin(), fr!(23));
        assert_eq!(mle.clone().eval_x(&v_01).eval_y(&v_11).fin(), fr!(24));

        assert_eq!(mle.clone().eval_x(&v_10).eval_y(&v_00).fin(), fr!(31));
        assert_eq!(mle.clone().eval_x(&v_10).eval_y(&v_01).fin(), fr!(32));
        assert_eq!(mle.clone().eval_x(&v_10).eval_y(&v_10).fin(), fr!(33));
        assert_eq!(mle.clone().eval_x(&v_10).eval_y(&v_11).fin(), fr!(34));

        assert_eq!(mle.clone().eval_x(&v_11).eval_y(&v_00).fin(), fr!(41));
        assert_eq!(mle.clone().eval_x(&v_11).eval_y(&v_01).fin(), fr!(42));
        assert_eq!(mle.clone().eval_x(&v_11).eval_y(&v_10).fin(), fr!(43));
        assert_eq!(mle.clone().eval_x(&v_11).eval_y(&v_11).fin(), fr!(44));
    }
}
