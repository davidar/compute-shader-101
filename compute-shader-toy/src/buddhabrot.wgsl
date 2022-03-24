type int = i32;
type uint = u32;
type float = f32;

type int2 = vec2<i32>;
type int3 = vec3<i32>;
type int4 = vec4<i32>;
type uint2 = vec2<u32>;
type uint3 = vec3<u32>;
type uint4 = vec4<u32>;
type float2 = vec2<f32>;
type float3 = vec3<f32>;
type float4 = vec4<f32>;

struct Params {
    frame: int;
};

struct AtomicStorageBuffer {
    data: array<atomic<i32>>;
};

struct FloatStorageBuffer {
    data: array<vec4<f32>>;
};

[[group(0), binding(0)]] var<uniform> params: Params;
[[group(0), binding(1)]] var col: texture_storage_2d<rgba16float,write>;
[[group(0), binding(2)]] var<storage,read_write> buf: AtomicStorageBuffer;
[[group(0), binding(3)]] var<storage,read_write> fbuf: FloatStorageBuffer;

// https://www.jcgt.org/published/0009/03/02/
// https://www.pcg-random.org/
fn pcg(seed: ptr<function, uint>) -> float {
	*seed = *seed * 747796405u + 2891336453u;
	let word = ((*seed >> ((*seed >> 28u) + 4u)) ^ *seed) * 277803737u;
	return float((word >> 22u) ^ word) / float(0xffffffffu);
}

fn smoothstep(edge0: float3, edge1: float3, x: float3) -> float3 {
    let t = clamp((x - edge0) / (edge1 - edge0), float3(0.0), float3(1.0));
    return t * t * (3.0 - 2.0 * t);
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_hist([[builtin(global_invocation_id)]] global_id: uint3) {
    let resolution = float2(textureDimensions(col));
    var seed = global_id.x + global_id.y * uint(resolution.x) + uint(params.frame) * uint(resolution.x * resolution.y);
    for (var iter = 0; iter < 8; iter = iter + 1) {
    let aspect = resolution.xy / resolution.y;
    let uv  = float2(float(global_id.x) + pcg(&seed), float(global_id.y) + pcg(&seed)) / resolution.xy;
    let uv0 = float2(float(global_id.x) + pcg(&seed), float(global_id.y) + pcg(&seed)) / resolution.xy;
    let c  = (uv  * 2. - 1.) * aspect * 1.5;
    let z0 = (uv0 * 2. - 1.) * aspect * 1.5;
    var z = z0;
    var n = 0;
    for (n = 0; n < 2500; n = n + 1) {
        z = float2(z.x * z.x - z.y * z.y, 2. * z.x * z.y) + c;
        if (dot(z,z) > 4.) { break; }
    }
    z = z0;
    for (var i = 0; i < 2500; i = i + 1) {
        z = float2(z.x * z.x - z.y * z.y, 2. * z.x * z.y) + c;
        if (dot(z,z) > 4.) { break; }
        let t = float(params.frame) / 60.;
        let p = (cos(.3*t) * z + sin(.3*t) * c) / 1.5 / aspect * .5 + .5;
        let id1 = int(resolution.x * p.x) + int(resolution.y * p.y) * int(resolution.x);
        let id2 = int(resolution.x * p.x) + int(resolution.y * (1. - p.y)) * int(resolution.x);
        if (n < 25) {
            atomicAdd(&buf.data[id1*4+2], 1);
            atomicAdd(&buf.data[id2*4+2], 1);
        } else if (n < 250) {
            atomicAdd(&buf.data[id1*4+1], 1);
            atomicAdd(&buf.data[id2*4+1], 1);
        } else if (n < 2500) {
            atomicAdd(&buf.data[id1*4+0], 1);
            atomicAdd(&buf.data[id2*4+0], 1);
        }
    }
    }
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_image([[builtin(global_invocation_id)]] global_id: uint3) {
    let resolution = float2(textureDimensions(col));
    let id = int(global_id.x) + int(global_id.y) * int(resolution.x);
    let x = float(atomicLoad(&buf.data[id*4+0]));
    let y = float(atomicLoad(&buf.data[id*4+1]));
    let z = float(atomicLoad(&buf.data[id*4+2]));
    var r = float3(x + y + z, y + z, z) / 3e3;
    r = smoothstep(float3(0.), float3(1.), 2.5 * pow(r, float3(.9, .8, .7)));
    textureStore(col, int2(global_id.xy), float4(r, 1.));
    atomicStore(&buf.data[id*4+0], int(x * .7));
    atomicStore(&buf.data[id*4+1], int(y * .7));
    atomicStore(&buf.data[id*4+2], int(z * .7));
}
