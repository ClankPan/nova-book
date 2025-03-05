use ark_ff::{AdditiveGroup, Field};
use ark_test_curves::bls12_381::Fr;
fn main() {
    // x^3 - 9x^2 + 26x - 24 = 0
    let coeffs = vec![fr!(-24), fr!(26), fr!(-9), fr!(1)];

    assert!(satisfy_poly(fr!(2), &coeffs));
    assert!(satisfy_poly(fr!(3), &coeffs));
    assert!(satisfy_poly(fr!(4), &coeffs));

    assert!(!satisfy_poly(fr!(5), &coeffs));
    assert!(!satisfy_poly(fr!(6), &coeffs));
    assert!(!satisfy_poly(fr!(7), &coeffs));
}

fn satisfy_poly(x: Fr, coeffs: &[Fr]) -> bool {
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