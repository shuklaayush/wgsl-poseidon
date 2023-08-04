@group(0)
@binding(0)
var<storage, read_write> buf_out: array<BigInt256>;

@group(0)
@binding(1)
var<storage, read> buf_in: array<BigInt256>;

// @group(0)
// @binding(2)
// var<uniform> n: u32;

// @group(0)
// @binding(2)
// var<storage, write> twiddles: array<BigInt256>;

// nth root of unity
fn omega(logn: u32) -> BigInt256 {
    switch logn {
        case 0u {
            return BigInt256(array<u32, 16>(0x0001u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u));
        }
        case 1u {
            return BigInt256(array<u32, 16>(0x0000u, 0xf000u, 0xf593u, 0x43e1u, 0x7091u, 0x79b9u, 0xe848u, 0x2833u, 0x585du, 0x8181u, 0x45b6u, 0xb850u, 0xa029u, 0xe131u, 0x4e72u, 0x3064u));
        }
        case 2u {
            return BigInt256(array<u32, 16>(0x3636u, 0x8f70u, 0x0470u, 0x2312u, 0x6becu, 0xfd73u, 0x24f6u, 0x5ceau, 0x4104u, 0x3fd8u, 0x6e19u, 0x048bu, 0xa029u, 0xe131u, 0x4e72u, 0x3064u));
        }
        case 3u {
            return BigInt256(array<u32, 16>(0x5e80u, 0xc1bdu, 0xad4au, 0x948du, 0x0a0au, 0xf817u, 0x7366u, 0x5262u, 0xef36u, 0x96afu, 0x9e2fu, 0xec9bu, 0x4f22u, 0xc8c1u, 0x7de1u, 0x2b33u));
        }
        case 4u {
            return BigInt256(array<u32, 16>(0x460bu, 0xe306u, 0x09c6u, 0xb115u, 0xfb98u, 0x174eu, 0xfbe1u, 0x996du, 0x508cu, 0x94ddu, 0x4f45u, 0x1c6eu, 0xbf4eu, 0x16cbu, 0x2ca2u, 0x2108u));
        }
        case 5u {
            return BigInt256(array<u32, 16>(0x12d0u, 0x3bb5u, 0x4c53u, 0x3eedu, 0xeb1du, 0x838eu, 0xd51bu, 0x9c18u, 0xb2a9u, 0x47c0u, 0x200du, 0x9678u, 0x93d2u, 0x306bu, 0x2c6u, 0x09c5u));
        }
        case 6u {
            return BigInt256(array<u32, 16>(0x023au, 0x118fu, 0xfb05u, 0xdb94u, 0x24beu, 0x26e3u, 0xcb24u, 0x46a6u, 0xadf2u, 0x49bdu, 0xdb76u, 0xc24cu, 0x0fcau, 0x5b08u, 0x144du, 0x1418u));
        }
        case 7u {
            return BigInt256(array<u32, 16>(0x1811u, 0xba9du, 0x470cu, 0x9d0eu, 0x4c79u, 0xb6f2u, 0x5564u, 0x1dcbu, 0x43e0u, 0xe859u, 0xe19cu, 0x6927u, 0x93b9u, 0xe8f3u, 0x01f9u, 0x13ddu));
        }
        default {
            return BigInt256(array<u32, 16>(0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u, 0x0000u));
        }
    }
}

fn reverse_bits(b: u32, len: u32) -> u32 {
    var num = b;
    var rev = 0u;
    for (var i = 0u; i < len; i += 1u) {
        rev = (rev << 1u) | (num & 1u);
        num >>= 1u;
    }
    return rev;
}

// @compute
// @workgroup_size(1)
// fn precompute(@builtin(global_invocation_id) global_id: vec3<u32>) {
//     var i = global_id.x;

//     // Compute twiddle factors
//     var w: BigInt256 = omega4();
//     twiddles[i] = fr_pow(&w, i);
// }

fn compute_twiddle_factors(n: u32, w: ptr<function, BigInt256>) -> array<BigInt256, 4u> {
    var twiddles: array<BigInt256, 4u>;
    twiddles[0] = fr_get_one();
    for(var i = 1u; i < n; i += 1u) {
        var prev = twiddles[i - 1u];
        twiddles[i] = fr_mul(&prev, w);
    }
    
    return twiddles;
}

// Size should be n
var<workgroup> buf_tmp: array<BigInt256, 8u>;

// Cooley-Tukey FFT algorithm
@compute
@workgroup_size(8) // Should be equal to n (number of coefficients in a polynomial)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    var n = 8u;
    var logn = 3u;
    var w: BigInt256 = omega(logn);

    var offset = n * workgroup_id.x;
    
    // Precompute twiddle factors
    // TODO: Move to separate kernel to remove redundancy
    var twiddles = compute_twiddle_factors(n, &w); 
    
    // TODO: Move to a separate kernel and reduce workgroup size of this to n/2 
    // Order based on reverse-bit
    var i_rev = reverse_bits(i, logn);
    buf_tmp[i] = buf_in[offset + i_rev];
    
    for(var s = 0u; s < logn; s += 1u) {
        // TODO: Do I need a storage barrier?
        // storageBarrier();
        workgroupBarrier();

        var m = 1u << s;
        var m2 = 1u << (s + 1u);
        var twiddle_div = n >> (s + 1u);

        var j = i & (m - 1u);
        var i2 = ((i / m) * m2) % n;
        // var i2 = (i / m) * m2;

        var t = buf_tmp[i2 + j];
        var u = buf_tmp[i2 + j + m];

        var twiddle = twiddles[j * twiddle_div];

        var v = fr_mul(&twiddle, &u);

        // Butterfly
        buf_tmp[i2 + j] = fr_add(&t, &v);
        buf_tmp[i2 + j + m] = fr_sub(&t, &v);
    }
    
    buf_out[offset + i] = buf_tmp[i];
}
