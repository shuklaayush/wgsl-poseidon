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

var<workgroup> buf_tmp: array<BigInt256, 4u>;

// 2^2 th root of unity
fn omega4() -> BigInt256 {
    var p: BigInt256;

    p.limbs[0] = 0x3636u;
    p.limbs[1] = 0x8f70u;
    p.limbs[2] = 0x0470u;
    p.limbs[3] = 0x2312u;
    p.limbs[4] = 0x6becu;
    p.limbs[5] = 0xfd73u;
    p.limbs[6] = 0x24f6u;
    p.limbs[7] = 0x5ceau;
    p.limbs[8] = 0x4104u;
    p.limbs[9] = 0x3fd8u;
    p.limbs[10] = 0x6e19u;
    p.limbs[11] = 0x048bu;
    p.limbs[12] = 0xa029u;
    p.limbs[13] = 0xe131u;
    p.limbs[14] = 0x4e72u;
    p.limbs[15] = 0x3064u;

    return p;
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

@compute
@workgroup_size(4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    var n = 4u;
    var logn = 2u;
    var w: BigInt256 = omega4();

    var offset = n * workgroup_id.x;
    
    // Precompute twiddle factors
    var twiddles = compute_twiddle_factors(n, &w); 
    
    // Order based on reverse-bit
    var i_rev = reverse_bits(i, logn);
    buf_tmp[i] = buf_in[offset + i_rev];
    
    for(var s = 0u; s < logn; s += 1u) {
        // TODO: Do I need a storage barrier?
        // storageBarrier();
        workgroupBarrier();

        var m = 1u << s;
        var k = i % m;
        var j = (i / 2u * m) * 2u * m + k;

        if (j + m < n) {
            var t = buf_tmp[j];
            var u = buf_tmp[j + m];

            var twiddle = twiddles[k * (n / (m * 2u))];

            var v = fr_mul(&twiddle, &u);

            // Butterfly
            buf_tmp[j] = fr_add(&t, &v);
            buf_tmp[j + m] = fr_sub(&t, &v);
        }
    }
    
    buf_out[offset + i] = buf_tmp[i];
}
