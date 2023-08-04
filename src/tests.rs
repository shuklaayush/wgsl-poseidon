use std::vec;

use crate::gpu::double_buffer_compute;
use crate::utils::{bigints_to_bytes, u32s_to_bigints};
use crate::wgsl::concat_files;
use ark_poly::DenseUVPolynomial;
use num_bigint::BigUint;
use stopwatch::Stopwatch;

use ark_bn254::Fr;
use ark_poly::{univariate::DensePolynomial, EvaluationDomain, GeneralEvaluationDomain};
// use ark_ff::{Zero, One};

#[test]
pub fn test_ntt() {
    let num_polys = 60000;
    let num_coeffs = 8;

    // 1, w, w^2, w^3
    let domain = GeneralEvaluationDomain::<Fr>::new(num_coeffs).unwrap();

    let mut rng = rand::thread_rng();

    let polys: Vec<DensePolynomial<Fr>> = (0..num_polys)
        .map(|_| DensePolynomial::<Fr>::rand(num_coeffs - 1, &mut rng))
        // .map(|_| DensePolynomial::<Fr>::from_coefficients_vec((0..num_coeffs).map(|i| Fr::from((num_coeffs - i) as u64) * Fr::one()).collect()))
        // .map(|_| DensePolynomial::<Fr>::from_coefficients_vec(vec![Fr::one(); num_coeffs]))
        .collect();
    let coeffs: Vec<Vec<BigUint>> = polys
        .iter()
        .map(|poly| poly.coeffs().iter().map(|&c| BigUint::from(c)).collect())
        .collect();

    let sw = Stopwatch::start_new();
    let expected: Vec<Vec<BigUint>> = polys
        .iter()
        .map(|poly| {
            domain
                .fft(&poly)
                .iter()
                .map(|&c| BigUint::from(c))
                .collect()
        })
        .collect();
    println!("CPU took {}ms", sw.elapsed_ms());

    let input_to_gpu = bigints_to_bytes(&coeffs.iter().flatten().collect());

    // Send to the GPU
    let wgsl = concat_files(vec![
        "src/wgsl/bigint.wgsl",
        "src/wgsl/fr.wgsl",
        "src/wgsl/ntt.wgsl",
        "src/wgsl/structs.wgsl",
    ]);

    let out_buf = vec![0u8; input_to_gpu.len()];

    // let sw = Stopwatch::start_new();
    let result = pollster::block_on(double_buffer_compute(
        &wgsl,
        &out_buf,
        &input_to_gpu,
        num_polys,
        num_coeffs,
    ))
    .unwrap();
    // println!("GPU took {}ms", sw.elapsed_ms());

    let result = u32s_to_bigints(result);
    // println!("result: {:?}", result);

    for i in 0..num_polys {
        for j in 0..num_coeffs {
            assert_eq!(result[i * num_coeffs + j], expected[i][j]);
        }
    }
}
