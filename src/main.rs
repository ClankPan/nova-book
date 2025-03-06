use ark_ff::{AdditiveGroup, Field};
use ark_test_curves::bls12_381::Fr;
use itertools::Itertools;

use std::{
    collections::HashMap,
    ops::{Add, Mul},
    vec,
};

fn main() {
    // MLE::eq("x",&bvec![0,1,1]);
    // vector_mle(&frvec![0,1,1]);
    let m1_mle = matrix_mle(&frmatrix!(
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ));
    let m2_mle = matrix_mle(&frmatrix!(
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ));
    let m3_mle = matrix_mle(&frmatrix!(
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ));

    let m = 4;
    let n = 8;

    // let x = vec![0,1,1,2,3,6,6];
    // let w = vec![];
    let z1_mle = vector_mle("y", &frvec![0, 1, 1, 2, 3, 6, 6, 1]);

    let m1z1 = m1_mle * z1_mle.clone();
    let m2z1 = m2_mle * z1_mle.clone();
    let m3z1 = m3_mle * z1_mle.clone();

    println!("\n");
    let sum_m1z1 = sum_all_patterns!(n, "y", m1z1);
    let sum_m2z1 = sum_all_patterns!(n, "y", m2z1);
    let sum_m3z1 = sum_all_patterns!(n, "y", m3z1);
    let g = sum_m1z1.clone() * sum_m2z1.clone() + fr!(-1) * sum_m3z1.clone();
    let sum_g = sum_all_patterns!(m, "x", g);
    println!("sum_g: {:?}", sum_g.fin().unwrap())
}

type ProdTerms = (Fr, Vec<bool>);
#[derive(Clone, Debug)]
pub struct MLE {
    sum: Vec<(Fr, HashMap<String, ProdTerms>)>,
    coeff: Fr,
}

pub fn vector_mle(variable: &str, vector: &[Fr]) -> MLE {
    // todo check vector len is pow of 2
    all_bit_patterns(vector.len())
        .into_iter()
        .enumerate()
        .filter_map(|(i, pattern)| {
            if vector[i] == Fr::ZERO {
                return None;
            }
            Some(vector[i] * MLE::eq(variable, &pattern))
        })
        .reduce(|acc, x| acc + x)
        .unwrap()
}

pub fn matrix_mle(matrix: &[Vec<Fr>]) -> MLE {
    let m = matrix.len();
    let n = matrix[0].len();
    // todo check m,n len are pow of 2
    all_bit_patterns(m)
        .into_iter()
        .enumerate()
        .filter_map(|(i, x_pattern)| {
            assert!(matrix[i].len() == n);
            all_bit_patterns(n)
                .into_iter()
                .enumerate()
                .filter_map(|(j, y_pattern)| {
                    if matrix[i][j] == Fr::ZERO {
                        return None;
                    }
                    Some(matrix[i][j] * MLE::eq("x", &x_pattern) * MLE::eq("y", &y_pattern))
                })
                .reduce(|acc, x| acc + x)
        })
        .reduce(|acc, x| acc + x)
        .unwrap()
}

impl MLE {
    pub fn eq(variable: &str, booleans: &[bool]) -> Self {
        let mut map = HashMap::new();
        map.insert(variable.to_string(), (fr!(1), booleans.to_vec()));
        Self {
            sum: vec![(fr!(1), map)],
            coeff: fr!(1),
        }
    }
    pub fn evaluate(self, variable: &str, v: &[bool]) -> Self {
        let mut sum = vec![];

        // println!("self.sum: {:?}", self.sum);

        for sum_term in self.sum {
            let (coeff, mut map) = sum_term;

            match map.remove(variable) {
                Some((v_coeff, v_prod_terms)) => {
                    // 全て一致していれば、このsum_termは0にはならないので、sumに再び加える。
                    assert!(v.len() == v_prod_terms.len());
                    if v.iter()
                        .zip(v_prod_terms.iter())
                        .all(|(v_i, t_i)| v_i == t_i)
                    {
                        sum.push((coeff * v_coeff, map)); //変数に値を入れて評価して得られた点は係数にまとめる。
                    }
                }
                None => {
                    sum.push((coeff, map));
                }
            }
        }

        Self {
            sum,
            coeff: self.coeff,
        }
    }

    pub fn fin(self) -> Result<Fr, String> {
        let mut sum = fr!(0);
        for (coeff, map) in self.sum {
            if !map.is_empty() {
                return Err(String::from("empty"));
            }
            sum += coeff
        }

        Ok(sum)
    }
}

// EqMLE * EqMLE
impl Mul for MLE {
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
                // let variables: Vec<String> = map_0.keys().cloned().into_iter().chain(map_1.keys().cloned().into_iter()).collect::<Vec<_>>().into_iter().sorted().collect();
                // println!("variables: {:?}", variables);
                let variables: Vec<String> = map_0
                    .keys()
                    .cloned()
                    .chain(map_1.keys().cloned())
                    .sorted()
                    .dedup()
                    .collect();
                for variable in variables {
                    // println!("key: {}", variable);
                    match (map_0.get(&variable), map_1.get(&variable)) {
                        (None, None) => panic!(),
                        (None, Some(v)) => {
                            new_map.insert(variable.to_string(), v.clone());
                        }
                        (Some(v), None) => {
                            new_map.insert(variable.to_string(), v.clone());
                        }
                        (Some(v0), Some(v1)) => {
                            // println!("key: {}", variable);
                            let (coeff_v0, prod_terms_v0) = v0;
                            let (coeff_v1, prod_terms_v1) = v1;
                            // println!("prod_terms_v0, prod_terms_v0: {:?}, {:?}", prod_terms_v0,prod_terms_v1);
                            // 掛け合わせたprod_termが0にならないかを調べる。
                            let mut new_prod_terms = vec![];
                            for (b, _b) in prod_terms_v0.iter().zip(prod_terms_v1) {
                                if b == _b {
                                    // println!("push");
                                    new_prod_terms.push(*b);
                                } else {
                                    // println!("break");
                                    // もし、評価した結果が0になるのなら、このsum_termは0になるので、次に移動する。
                                    continue 'outer;
                                }
                            }
                            // println!("all same");
                            new_map.insert(
                                variable.to_string(),
                                (coeff_v0 * coeff_v1, new_prod_terms),
                            );
                        }
                    };
                }

                sum.push((coeff_0 * coeff_1, new_map))
            }
        }

        MLE {
            sum,
            coeff: self.coeff * rhs.coeff,
        }
    }
}

impl Add for MLE {
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
impl Mul<Fr> for MLE {
    type Output = Self;

    fn mul(mut self, scalar: Fr) -> Self::Output {
        self.coeff *= scalar;
        self
    }
}

// Fr * EqMLE
impl Mul<MLE> for Fr {
    type Output = MLE;

    fn mul(self, mut rhs: MLE) -> Self::Output {
        rhs.coeff *= self;
        rhs
    }
}

fn all_bit_patterns(n: usize) -> Vec<Vec<bool>> {
    let b_len = n.trailing_zeros() as usize;
    // println!("bits: {b_len}");
    let mut result = Vec::with_capacity(b_len);

    for i in 0..n {
        let mut bits = Vec::with_capacity(n);
        for b in (0..b_len).rev() {
            // b番目のビットが1なら true, 0なら false
            if (i >> b) & 1 == 1 {
                bits.push(true);
            } else {
                bits.push(false);
            }
        }
        result.push(bits);
    }
    // println!("all_bit_patterns {:?}", result);
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

#[macro_export]
macro_rules! bvec {
    // パターン: カンマ区切りの式 (expr) 列
    ($($val:expr),* $(,)?) => {
        {
            let mut temp = Vec::new();
            $(
                let num = $val;
                // 0か1以外ならアサート失敗
                assert!(
                    num == 0 || num == 1,
                    "boolvec!: invalid value `{}`, allowed only 0 or 1", num
                );
                // 0→false, 1→true としてpush
                temp.push(num == 1);
            )*
            temp
        }
    };
}
#[macro_export]
macro_rules! frvec {
    // パターン: カンマ区切りの式 (expr) 列
    ($($val:expr),* $(,)?) => {
        {
            let mut temp = Vec::new();
            $(
                let num = $val;
                temp.push(Fr::from(num));
            )*
            temp
        }
    };
}

#[macro_export]
macro_rules! bmatrix {
    //
    // 複数の [ ... ] (それぞれ1行) を並べる形
    (
        $(
            [ $($val:expr),* $(,)? ]
        ),* $(,)?
    ) => {{
        let mut outer = Vec::new();

        // まだ列数が確定していないので Option<usize> を使う
        let mut expected_cols: Option<usize> = None;

        $(
            let mut row_vec = Vec::new();
            let mut col_count = 0;

            // 各要素に対してチェック
            $(
                let num = $val;
                row_vec.push(Fr::from(num));
                col_count += 1;
            )*

            // 初回行で列数を確定し、それ以降は照合する
            match expected_cols {
                Some(c) => {
                    // もし今回の行が c 列と異なるならエラー
                    assert!(
                        col_count == c,
                        "bmat!: row has length {}, but previous row(s) had length {}",
                        col_count, c
                    );
                },
                None => {
                    // まだ列数が未定 => 今回の列数を採用
                    expected_cols = Some(col_count);
                }
            }

            outer.push(row_vec);
        )*

        // 結果: Vec<Vec<bool>>
        outer
    }};
}
#[macro_export]
macro_rules! frmatrix {
    (
        $(
            [ $($val:expr),* $(,)? ]
        ),* $(,)?
    ) => {{
        let mut outer = Vec::new();
        // 列数がまだ未定 → Option
        let mut expected_cols: Option<usize> = None;

        // 行ループ用カウンタ（メッセージ表示に使う）
        let mut row_idx = 0_usize;

        $(
            let mut row_vec = Vec::new();
            let mut col_count = 0;

            // 各セルをチェック・変換
            $(
                let num = $val;
                assert!(
                    num == 0 || num == 1,
                    "frmatrix!: invalid value `{}`, allowed only 0 or 1",
                    num
                );
                row_vec.push(
                    if num == 1 {
                        ark_test_curves::bls12_381::Fr::ONE
                    } else {
                        ark_test_curves::bls12_381::Fr::ZERO
                    }
                );
                col_count += 1;
            )*

            // 列数チェック
            match expected_cols {
                Some(c) => {
                    assert!(
                        col_count == c,
                        "frmatrix!: row {} has length {}, but expected {} cols",
                        row_idx, col_count, c
                    );
                },
                None => {
                    // 初回行で列数確定
                    expected_cols = Some(col_count);
                }
            }

            outer.push(row_vec);
            row_idx += 1;
        )*

        // もし expected_cols が None なら行が0件だった
        // 一応読んでおけば unused にならない & 意図的に許容する
        if let Some(cols) = expected_cols {
            // 行数 row_idx, 列数 cols が最終的に確定した形
            // 必要ならここでさらに assert!してもよい
            let _ = (row_idx, cols); // 例: ここで何か使うなら
        }

        outer // => Vec<Vec<ark_test_curves::bls12_381::Fr>>
    }};
}

#[macro_export]
macro_rules! sum_all_patterns {
    // 第1引数: n (usize など),
    // 第2引数: 変数名 (例: "y"),
    // 第3引数: m1_mle
    ($n:expr, $var:expr, $m:expr) => {{
        // all_bit_patterns($n) は事前に定義されていると仮定
        all_bit_patterns($n)
            .into_iter()
            .map(|pattern| $m.clone().evaluate($var, &pattern))
            .reduce(|acc, x| acc + x)
            .unwrap()
    }};
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

    #[test]
    fn check_mle() {
        let _ = fr!(5) * MLE::eq("x", &[true, true]) * MLE::eq("x", &[true, false]);

        let f_mle = fr!(22) * MLE::eq("x", &[true, true])
            + fr!(33) * MLE::eq("x", &[true, false])
            + fr!(44) * MLE::eq("x", &[false, true])
            + fr!(55) * MLE::eq("x", &[false, false]);

        assert_eq!(
            f_mle.clone().evaluate("x", &[true, true]).fin().unwrap(),
            fr!(22)
        );
        assert_eq!(
            f_mle.clone().evaluate("x", &[true, false]).fin().unwrap(),
            fr!(33)
        );
        assert_eq!(
            f_mle.clone().evaluate("x", &[false, true]).fin().unwrap(),
            fr!(44)
        );
        assert_eq!(
            f_mle.clone().evaluate("x", &[false, false]).fin().unwrap(),
            fr!(55)
        );

        let f_mle = MLE::eq("x", &[true, true]) * MLE::eq("y", &[true, true]);
        assert_eq!(
            f_mle
                .clone()
                .evaluate("x", &[true, true])
                .evaluate("y", &[true, true])
                .fin()
                .unwrap(),
            fr!(1)
        );
        assert_eq!(
            f_mle
                .clone()
                .evaluate("y", &[true, true])
                .evaluate("x", &[true, true])
                .fin()
                .unwrap(),
            fr!(1)
        );

        assert_eq!(
            f_mle
                .clone()
                .evaluate("x", &[false, true])
                .evaluate("y", &[true, true])
                .fin()
                .unwrap(),
            fr!(0)
        );
        assert_eq!(
            f_mle
                .clone()
                .evaluate("x", &[true, false])
                .evaluate("y", &[true, true])
                .fin()
                .unwrap(),
            fr!(0)
        );
        assert_eq!(
            f_mle
                .clone()
                .evaluate("x", &[false, false])
                .evaluate("y", &[true, true])
                .fin()
                .unwrap(),
            fr!(0)
        );
        assert_eq!(
            f_mle
                .clone()
                .evaluate("x", &[false, true])
                .evaluate("y", &[false, true])
                .fin()
                .unwrap(),
            fr!(0)
        );
        assert_eq!(
            f_mle
                .clone()
                .evaluate("x", &[true, false])
                .evaluate("y", &[true, false])
                .fin()
                .unwrap(),
            fr!(0)
        );
        assert_eq!(
            f_mle
                .clone()
                .evaluate("x", &[false, false])
                .evaluate("y", &[false, false])
                .fin()
                .unwrap(),
            fr!(0)
        );

        let f_mle = MLE::eq("x", &[true, true]) + MLE::eq("y", &[true, true]);
        assert_eq!(
            f_mle
                .clone()
                .evaluate("x", &[true, true])
                .evaluate("y", &[true, true])
                .fin()
                .unwrap(),
            fr!(2)
        );
    }

    #[test]
    fn check_vector_matrix_mle() {
        let z_mle = vector_mle("x", &[fr!(11), fr!(22), fr!(33), fr!(44)]);
        assert_eq!(
            z_mle.clone().evaluate("x", &[false, false]).fin().unwrap(),
            fr!(11)
        );
        assert_eq!(
            z_mle.clone().evaluate("x", &[false, true]).fin().unwrap(),
            fr!(22)
        );
        assert_eq!(
            z_mle.clone().evaluate("x", &[true, false]).fin().unwrap(),
            fr!(33)
        );
        assert_eq!(
            z_mle.clone().evaluate("x", &[true, true]).fin().unwrap(),
            fr!(44)
        );

        let matrix = vec![
            vec![fr!(11), fr!(21), fr!(31), fr!(41)],
            vec![fr!(12), fr!(22), fr!(32), fr!(42)],
            vec![fr!(13), fr!(23), fr!(33), fr!(43)],
            vec![fr!(14), fr!(24), fr!(34), fr!(44)],
        ];
        let m_mle = matrix_mle(&matrix);
        assert_eq!(
            m_mle
                .clone()
                .evaluate("x", &[false, false])
                .evaluate("y", &[false, false])
                .fin()
                .unwrap(),
            fr!(11)
        );
        assert_eq!(
            m_mle
                .clone()
                .evaluate("x", &[false, false])
                .evaluate("y", &[false, true])
                .fin()
                .unwrap(),
            fr!(21)
        );
        assert_eq!(
            m_mle
                .clone()
                .evaluate("x", &[false, false])
                .evaluate("y", &[true, false])
                .fin()
                .unwrap(),
            fr!(31)
        );
        assert_eq!(
            m_mle
                .clone()
                .evaluate("x", &[false, false])
                .evaluate("y", &[true, true])
                .fin()
                .unwrap(),
            fr!(41)
        );
        assert_eq!(
            m_mle
                .clone()
                .evaluate("x", &[false, true])
                .evaluate("y", &[false, false])
                .fin()
                .unwrap(),
            fr!(12)
        );
        assert_eq!(
            m_mle
                .clone()
                .evaluate("x", &[false, true])
                .evaluate("y", &[false, true])
                .fin()
                .unwrap(),
            fr!(22)
        );
    }
}
