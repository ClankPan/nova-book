use ark_ff::{AdditiveGroup, Field};
// use ark_test_curves::bls12_381::Fr;



use std::{
    ops::{Add, AddAssign, Mul, MulAssign},
    vec,
};

use crate::all_bit_patterns;


#[cfg(not(test))]
use ark_test_curves::bls12_381::Fr;
#[cfg(test)]
type Fr = crate::fp101::Fp101;

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

pub struct MLE {}

impl MLE {
    pub fn vec_x(vector: &[Fr]) -> Poly {
        vector_mle(vector, Var::X)
    }
    pub fn vec_y(vector: &[Fr]) -> Poly {
        vector_mle(vector, Var::Y)
    }
    pub fn matrix(matrix: &Vec<Vec<Fr>>) -> Poly {
        matrix_mle(matrix)
    }
}

fn vector_mle(vector: &[Fr], var: Var) -> Poly {
    assert!(vector.len().is_power_of_two());
    assert!(vector.len() > 1);

    let eq = match var {
        Var::X => Eq::x,
        Var::Y => Eq::y,
    };

    let mut mle = Poly::new();
    for (i, pattern) in all_bit_patterns(vector.len()).into_iter().enumerate() {
        if vector[i] == Fr::ZERO {
            continue;
        }
        mle += vector[i] * eq(&bit_patterns_to_fr(&pattern));
    }

    mle
}

fn matrix_mle(matrix: &Vec<Vec<Fr>>) -> Poly {
    let m = matrix.len();
    let n = matrix[0].len();

    assert!(m.is_power_of_two());
    assert!(n.is_power_of_two());
    assert!(m > 1);
    assert!(n > 1);

    let mut mle = Poly::new();
    for (i, x_pattern) in all_bit_patterns(m).into_iter().enumerate() {
        for (j, y_pattern) in all_bit_patterns(n).into_iter().enumerate() {
            assert!(matrix[i].len() == n);
            if matrix[i][j] == Fr::ZERO {
                continue;
            }
            mle += matrix[i][j]
                * Eq::x(&bit_patterns_to_fr(&x_pattern))
                * Eq::y(&bit_patterns_to_fr(&y_pattern));
        }
    }

    mle
}

impl Default for Poly {
    fn default() -> Self {
        Self::new()
    }
}

pub fn bit_patterns_to_variables(pattern: &Vec<bool>) -> Vec<Option<Fr>> {
    pattern.iter().map(|b| Some(Fr::from(*b))).collect()
}

pub fn bit_patterns_to_fr(pattern: &Vec<bool>) -> Vec<Fr> {
    pattern.iter().map(|b| Fr::from(*b)).collect()
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
        assert!(!self.sum.is_empty());
        for (coeff, eq) in self.sum {
            assert!(eq.inners_x.is_none() & eq.inners_y.is_none()); // eqには係数しかないことを確認
            sum += coeff;
        }

        sum
    }

    pub fn sum(self, bit_len: usize, var: &Var) -> Poly {
        let mut sum = Poly::new();
        for pattern in all_bit_patterns(bit_len) {
            let pattern = bit_patterns_to_variables(&pattern);
            sum += self.clone().eval(&pattern, var);
        }

        sum
    }

    fn eval(self, evals: &Vec<Option<Fr>>, var: &Var) -> Self {
        let mut sum = vec![];

        for (mut coeff, eq) in self.sum {
            // 選択された変数の方に対して処理を行っていく。
            let (inners, _inners) = match var {
                Var::X => (eq.inners_x, eq.inners_y),
                Var::Y => (eq.inners_y, eq.inners_x),
            };

            let Some(inners) = inners else {
                panic!();
            };

            let mut new_inners = vec![];
            for inner in inners {
                assert!(inner.len() == evals.len());

                let mut new_inner = vec![];
                for (vi, ei) in inner.into_iter().zip(evals) {
                    // 評価して係数へまめる、もしくはそのまま変数を残す。
                    if let Some(ei) = ei {
                        let res = (Fr::ONE - vi) * (Fr::ONE - ei) + vi * ei;
                        // println!("vi: {}, ei: {}, res: {}\n", vi, ei, res);
                        coeff *= res;
                    } else {
                        new_inner.push(vi);
                    }
                }
                if !new_inner.is_empty() {
                    new_inners.push(new_inner)
                }
            }

            // println!("coeff: {}\n", coeff);

            if coeff == Fr::ZERO {
                continue;
            }

            let new_inners = if new_inners.is_empty() {
                None
            } else {
                Some(new_inners)
            };

            let (inners_x, inners_y) = match var {
                Var::X => (new_inners, _inners),
                Var::Y => (_inners, new_inners),
            };
            let eq = Eq { inners_x, inners_y };

            sum.push((coeff, eq))
        }
        Self {
            coeff: self.coeff,
            sum,
        }
    }

    pub fn eval_x(self, values: &Vec<Option<Fr>>) -> Self {
        self.eval(values, &Var::X)
    }

    pub fn eval_y(self, values: &Vec<Option<Fr>>) -> Self {
        self.eval(values, &Var::Y)
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
            let eq = term * eq.clone();
            if coeff == Fr::ZERO {
                // coeffが0の時はsumから除外する。
                continue;
            }
            sum.push((coeff, eq))
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
        let sum = match (self.sum.len(), rhs.sum.len()) {
            (0,_)=> {
                rhs.sum
            }
            (_, 0) => {
                self.sum
            }
            _ => {
                let mut sum = vec![];
                for (coeff_a, eq_a) in self.sum {
                    'inner: for (coeff_b, eq_b) in &rhs.sum {
                        let eq_ab = eq_a.clone() * eq_b.clone();
                        let coeff_ab = coeff_a * coeff_b;
                        if coeff_ab == Fr::ZERO {
                            // coeffが0の時はsumから除外する。
                            continue 'inner;
                        }
                        sum.push((coeff_ab, eq_ab))
                    }
                }
                sum
            }
        };
        Self { coeff, sum }
    }
}

// Poly *= Poly
impl MulAssign for Poly {
    fn mul_assign(&mut self, rhs: Self) {
        self.coeff = self.coeff * rhs.coeff;
        self.sum = match (self.sum.len(), rhs.sum.len()) {
            (0,_)=> {
                rhs.sum
            }
            (_, 0) => {
                return;
            }
            _ => {
                let mut sum = vec![];
                for (coeff_a, eq_a) in &self.sum {
                    'inner: for (coeff_b, eq_b) in &rhs.sum {
                        let eq_ab = eq_a.clone() * eq_b.clone();
                        let coeff_ab = coeff_a * coeff_b;
                        if coeff_ab == Fr::ZERO {
                            continue 'inner;
                        }
                        sum.push((coeff_ab, eq_ab))
                    }
                }
                sum
            }
        };
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
// Poly += Poly
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


type Vi = Vec<Fr>;
#[derive(PartialEq, Debug, Clone)]
pub struct Eq {
    // Noneの場合は評価済みを表す。
    inners_x: Option<Vec<Vi>>,
    inners_y: Option<Vec<Vi>>,
}

impl Eq {
    pub fn x(inner: &[Fr]) -> Self {
        Self {
            inners_x: Some(vec![inner.to_vec()]),
            inners_y: None,
        }
    }
    pub fn y(inner: &[Fr]) -> Self {
        Self {
            inners_x: None,
            inners_y: Some(vec![inner.to_vec()]),
        }
    }
}

// Eq * Eq -> Poly
impl Mul<Eq> for Eq {
    type Output = Eq;

    fn mul(self, rhs: Eq) -> Self::Output {
        fn merge_inner(inner_a: Option<Vec<Vi>>, inner_b: Option<Vec<Vi>>) -> Option<Vec<Vi>> {
            match (inner_a, inner_b) {
                (None, None) => None,
                (None, Some(inner)) => Some(inner),
                (Some(inner), None) => Some(inner),
                // todo: 同じやつはまとめてもいいかも。
                (Some(a), Some(b)) => Some(a.into_iter().chain(b).collect()),
            }
        }

        let inners_x = merge_inner(self.inners_x, rhs.inners_x);
        let inners_y = merge_inner(self.inners_y, rhs.inners_y);

        Self { inners_x, inners_y }
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

        let v_00 = bvec![0, 0].into_iter().map(|b| Some(Fr::from(b))).collect();
        let v_01 = bvec![0, 1].into_iter().map(|b| Some(Fr::from(b))).collect();
        let v_10 = bvec![1, 0].into_iter().map(|b| Some(Fr::from(b))).collect();
        let v_11 = bvec![1, 1].into_iter().map(|b| Some(Fr::from(b))).collect();

        assert_eq!(mle.clone().eval_x(&v_00).fin(), fr!(11));
        assert_eq!(mle.clone().eval_x(&v_01).fin(), fr!(22));
        assert_eq!(mle.clone().eval_x(&v_10).fin(), fr!(33));
        assert_eq!(mle.clone().eval_x(&v_11).fin(), fr!(44));
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

    #[test]
    fn check_fibobacci_ccs() {
        let v_00: Vec<_> = bvec![0, 0].into_iter().map(|b| Some(Fr::from(b))).collect();
        // let v_01: Vec<_> = bvec![0, 1].into_iter().map(|b| Some(Fr::from(b))).collect();
        // let v_10: Vec<_> = bvec![1, 0].into_iter().map(|b| Some(Fr::from(b))).collect();
        // let v_11: Vec<_> = bvec![1, 1].into_iter().map(|b| Some(Fr::from(b))).collect();

        // let v_000: Vec<_> = bvec![0, 0, 0].into_iter().map(|b| Some(Fr::from(b))).collect();
        let v_001: Vec<_> = bvec![0, 0, 1]
            .into_iter()
            .map(|b| Some(Fr::from(b)))
            .collect();
        // let v_010: Vec<_> = bvec![0, 1, 0].into_iter().map(|b| Some(Fr::from(b))).collect();
        // let v_011: Vec<_> = bvec![0, 1, 1].into_iter().map(|b| Some(Fr::from(b))).collect();
        // let v_100: Vec<_> = bvec![1, 0, 0].into_iter().map(|b| Some(Fr::from(b))).collect();
        // let v_101: Vec<_> = bvec![1, 0, 1].into_iter().map(|b| Some(Fr::from(b))).collect();
        // let v_110: Vec<_> = bvec![1, 1, 0].into_iter().map(|b| Some(Fr::from(b))).collect();
        // let v_111: Vec<_> = bvec![1, 1, 1].into_iter().map(|b| Some(Fr::from(b))).collect();

        let m1_mle = MLE::matrix(&frmatrix!(
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ));
        let m2_mle = MLE::matrix(&frmatrix!(
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ));
        let m3_mle = MLE::matrix(&frmatrix!(
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ));

        let m = 4;
        let n = 8;

        let z1_mle = MLE::vec_y(&frvec![0, 1, 1, 2, 3, 6, 6, 1]);

        let m1z1 = m1_mle * z1_mle.clone();
        let m2z1 = m2_mle * z1_mle.clone();
        let m3z1 = m3_mle * z1_mle.clone();

        println!("{:?}", m1z1.clone());
        println!("\n");
        println!("{:?}", m1z1.clone().eval_x(&v_00));
        println!("\n");
        println!("{:?}", m1z1.clone().eval_y(&v_001));
        println!("\n");
        println!("{:?}", m1z1.clone().eval_x(&v_00).eval_y(&v_001));

        assert_eq!(m1z1.clone().eval_y(&v_001).eval_x(&v_00).fin(), fr!(1));

        let g = m1z1.sum(n, &Var::Y) * m2z1.sum(n, &Var::Y) + fr!(-1) * m3z1.sum(n, &Var::Y);

        let sum_g = g.clone().sum(m, &Var::X).fin();
        assert_eq!(sum_g, Fr::ZERO);

        let mut h = Poly::new();
        for pattern in all_bit_patterns(m) {
            let variables = bit_patterns_to_variables(&pattern);
            h += g.clone().eval_x(&variables) * Eq::x(&bit_patterns_to_fr(&pattern))
        }

        let sum_h = h.sum(m, &Var::X).fin();
        assert_eq!(sum_h, Fr::ZERO);
    }

    #[test]
    fn check_fr_mul_poly() {
        let poly = fr!(11) * Poly::new();
        assert_eq!(fr!(11), poly.coeff);
    }

    #[test]
    fn check_non_binary_eval() {
        let r2 = fr!(84);
        let r1 = fr!(31);

        let mle = MLE::vec_x(&frvec![1, 1]);
        let res = mle.eval_x(&vec![Some(r1)]);
        assert_eq!(res.fin(), fr!(1-84)+fr!(84));


        let mle = MLE::vec_x(&frvec![1, 1, 0, 0]);
        println!("{:?}", mle);
        let mle = mle.eval_x(&vec![Some(r1), None]);
        println!("{:?}", mle);
        let mle = mle.eval_x(&vec![Some(r2)]);
        println!("{:?}", mle);
        assert_eq!(mle.fin(), fr!(-83));
    }

    #[test]
    fn check_t1() {

        let r1 = fr!(23);
        let r2 = fr!(48);

        let m1 = MLE::matrix(&frmatrix!(
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ));
        let z1 = MLE::vec_y(&frvec![0, 1, 1, 2, 3, 6, 6, 1]);
        let t1 = (m1.clone().eval_x(&vec![Some(r1), Some(r2)]) * z1.clone()).sum(8, &Var::Y);

        assert_eq!(fr!(17), t1.fin())
    }

    #[test]
    fn check_sumcheck() {
        let m1 = MLE::matrix(&frmatrix!(
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ));
        let m2 = MLE::matrix(&frmatrix!(
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ));
        let m3 = MLE::matrix(&frmatrix!(
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ));

        let m = 4;
        let n = 8;

        let z1 = MLE::vec_y(&frvec![0, 1, 1, 2, 3, 6, 6, 1]);

        let m1z1 = m1.clone() * z1.clone();
        let m2z1 = m2.clone() * z1.clone();
        let m3z1 = m3.clone() * z1.clone();

        // println!("m1z1: {:?}\n", m1z1);
        // println!("m2z1: {:?}\n", m2z1);
        // println!("m3z1: {:?}\n", m3z1);

        let sum_m1z1 = m1z1.clone().sum(n, &Var::Y);
        let sum_m2z1 = m2z1.clone().sum(n, &Var::Y);
        let sum_m3z1 = m3z1.clone().sum(n, &Var::Y);

        // println!("sum_m1z1: {:?}\n", sum_m1z1);
        // println!("sum_m2z1: {:?}\n", sum_m2z1);
        // println!("sum_m3z1: {:?}\n", sum_m3z1);

        let g = sum_m1z1 * sum_m2z1 + fr!(-1) * sum_m3z1;



        // 本来は乱数。
        let alpha = fr!(57); 
        let beta = frvec![9, 61];
        let r1 = fr!(23);
        let r2 = fr!(48);
        let r11 = fr!(25);
        let r22 = fr!(50);
        let r33 = fr!(49);

        let q = g.clone() * Eq::x(&beta); // Eq::x(&beta)はbetaが{0,1}でないなら高確率で0以外になるので、gがどの値でも0になることを確認できる。
        // println!("g: {:?}\n", g);
        // println!("eq: {:?}\n", Eq::x(&beta));
        let sum_q = q.clone().sum(m, &Var::X).fin();
        assert_eq!(sum_q, Fr::ZERO);

        /*-- Outer-sumcheck --*/

        /* Round 1 */
        // Prover
        let q1 = q.clone().eval_x(&vec![None, Some(Fr::ONE)]); // Proverが用意する。
        let s1 = q1.clone(); // Verifierに渡す。
        //  Verifier
        let sum_s1 = (s1.clone().eval_x(&vec![Some(Fr::ZERO)])
            + s1.clone().eval_x(&vec![Some(Fr::ONE)]))
        .fin();
        assert_eq!(sum_s1, Fr::ZERO);

        /* Round 2 */
        // Prover
        let q2 = q.clone().eval_x(&vec![Some(r1), None]);
        let s2 = q2.clone(); // Verifierに渡す。
        // Verifier
        let r1_s1 = s1.clone().eval_x(&vec![Some(r1)]).fin();
        let sum_s2 = (s2.clone().eval_x(&vec![Some(Fr::ZERO)])
            + s2.clone().eval_x(&vec![Some(Fr::ONE)]))
        .fin();
        // println!("{:?} {:?}", r1_s1, sum_s2);
        assert_eq!(r1_s1, sum_s2);
        assert!(r1_s1 != Fr::ZERO);
        let _r2_q2 = s2.clone().eval_x(&vec![Some(r2)]).fin();

        /*-- Batchin inner-sumcheck --*/
        let t1 = (m1.clone().eval_x(&vec![Some(r1), Some(r2)]) * z1.clone()).sum(n, &Var::Y);
        let t2 = (m2.clone().eval_x(&vec![Some(r1), Some(r2)]) * z1.clone()).sum(n, &Var::Y);
        let t3 = (m3.clone().eval_x(&vec![Some(r1), Some(r2)]) * z1.clone()).sum(n, &Var::Y);

        assert_eq!(fr!(17), t1.clone().fin());
        assert_eq!(fr!(11), t2.clone().fin());
        assert_eq!(fr!(29), t3.clone().fin());

        let t = t1 + alpha * t2 + alpha * alpha * t3;
        assert_eq!(fr!(26), t.clone().fin());

        /* Round 1 */
        // Prover
        let mut f1 = Poly::new();
        for pattern in all_bit_patterns(4) {
            f1 += (m1.clone().eval_x(&vec![Some(r1), Some(r2)]) * z1.clone()).eval_y(&vec![None, Some(fr!(pattern[0])), Some(fr!(pattern[1]))]);
        }
        for pattern in all_bit_patterns(4) {
            f1 += alpha * (m2.clone().eval_x(&vec![Some(r1), Some(r2)]) * z1.clone()).eval_y(&vec![None, Some(fr!(pattern[0])), Some(fr!(pattern[1]))]);
        }
        let alpha_pow2 = alpha * alpha;
        for pattern in all_bit_patterns(4) {
            f1 += alpha_pow2 * (m3.clone().eval_x(&vec![Some(r1), Some(r2)]) * z1.clone()).eval_y(&vec![None, Some(fr!(pattern[0])), Some(fr!(pattern[1]))]);
        }
        let q1 = f1;
        // Verifier
        let sum_q1 = q1.clone().eval_y(&vec![Some(Fr::ZERO)]) + q1.clone().eval_y(&vec![Some(Fr::ONE)]);
        assert_eq!(t.fin(), sum_q1.fin());

        /* Round 2 */
        // Prover
        let mut f2 = Poly::new();
        for pattern in all_bit_patterns(2) {
            f2 += (m1.clone().eval_x(&vec![Some(r1), Some(r2)]) * z1.clone()).eval_y(&vec![Some(r11), None, Some(fr!(pattern[0]))]);
        }
        for pattern in all_bit_patterns(2) {
            f2 += alpha * (m2.clone().eval_x(&vec![Some(r1), Some(r2)]) * z1.clone()).eval_y(&vec![Some(r11), None, Some(fr!(pattern[0]))]);
        }
        let alpha_pow2 = alpha * alpha;
        for pattern in all_bit_patterns(2) {
            f2 += alpha_pow2 * (m3.clone().eval_x(&vec![Some(r1), Some(r2)]) * z1.clone()).eval_y(&vec![Some(r11), None, Some(fr!(pattern[0]))]);
        }
        let q2 = f2;
        // Verifier
        let sum_q2 = q2.clone().eval_y(&vec![Some(Fr::ZERO)]) + q2.clone().eval_y(&vec![Some(Fr::ONE)]);
        let r11_q1 = q1.eval_y(&vec![Some(r11)]);
        assert_eq!(r11_q1.fin(), sum_q2.fin());
        
        /* Round 3 */
        // Prover
        let mut f3 = Poly::new();
        f3 += (m1.clone().eval_x(&vec![Some(r1), Some(r2)]) * z1.clone()).eval_y(&vec![Some(r11), Some(r22), None]);
        f3 += alpha * (m2.clone().eval_x(&vec![Some(r1), Some(r2)]) * z1.clone()).eval_y(&vec![Some(r11), Some(r22), None]);
        f3 += alpha_pow2 * (m3.clone().eval_x(&vec![Some(r1), Some(r2)]) * z1.clone()).eval_y(&vec![Some(r11), Some(r22), None]);
        let q3 = f3;
        // Verifier
        let sum_q3 = q3.clone().eval_y(&vec![Some(Fr::ZERO)]) + q3.clone().eval_y(&vec![Some(Fr::ONE)]);
        let r22_q2 = q2.eval_y(&vec![Some(r22)]);
        assert_eq!(r22_q2.fin(), sum_q3.fin());

        let r_t = m1z1.eval_x(&vec![Some(r1), Some(r2)]).eval_y(&vec![Some(r11), Some(r22), Some(r33)])
            + alpha * m2z1.eval_x(&vec![Some(r1), Some(r2)]).eval_y(&vec![Some(r11), Some(r22), Some(r33)])
            + alpha_pow2 * m3z1.eval_x(&vec![Some(r1), Some(r2)]).eval_y(&vec![Some(r11), Some(r22), Some(r33)]);
        let r33_q3 = q3.eval_y(&vec![Some(r33)]);
        assert_eq!(r33_q3.fin(), r_t.fin());

        /*-- Final check --*/


    }
}
