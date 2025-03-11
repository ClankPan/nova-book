use ark_ff::{Fp64, MontBackend, MontConfig};

#[derive(MontConfig)]
#[modulus = "101"]
#[generator = "2"]
pub struct F101Config;

pub type Fp101 = Fp64<MontBackend<F101Config, 1>>;