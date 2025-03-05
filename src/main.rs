use ark_ff::{AdditiveGroup, Field};
use ark_test_curves::bls12_381::Fr;

fn main() {

}

/// 二つの{0,1}の配列を受け取って、等しければ1を、異なれば0を返す。
pub fn eq_mle(x1: &[Fr], x2: &[Fr]) -> Fr {
    assert_eq!(x1.len(), x2.len());

    let mut prod = Fr::ONE;
    for (x1_i, x2_i) in x1.iter().zip(x2) {
        prod *= (Fr::ONE - x1_i) * (Fr::ONE - x2_i) + x1_i * x2_i
    }

    prod
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

    #[test]
    fn check_eq_mle() {
        assert_eq!(eq_mle(&vec![fr!(1)], &vec![fr!(1)]), Fr::ONE);
        assert_eq!(eq_mle(&vec![fr!(1)], &vec![fr!(0)]), Fr::ZERO);
        assert_eq!(eq_mle(&vec![fr!(0)], &vec![fr!(1)]), Fr::ZERO);
        assert_eq!(eq_mle(&vec![fr!(0)], &vec![fr!(0)]), Fr::ONE);
    }
}