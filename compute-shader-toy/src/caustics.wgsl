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

fn bufferStore(pos: int2, index: int, value: float4) {
    let dim = textureDimensions(col);
    fbuf.data[pos.x + pos.y * dim.x + index * dim.x * dim.y] = value;
}

fn bufferLoad(pos: int2, index: int) -> float4 {
    let dim = textureDimensions(col);
    return fbuf.data[pos.x + pos.y * dim.x + index * dim.x * dim.y];
}

fn bufferSample(uv: float2, index: int) -> float4 {
    let resolution = float2(textureDimensions(col));
    let p = clamp(uv, float2(0.), float2(1.)) * resolution - 0.5;
    let a = mix(bufferLoad(int2(p) + int2(0,0), index), bufferLoad(int2(p) + int2(1,0), index), fract(p.x));
    let b = mix(bufferLoad(int2(p) + int2(0,1), index), bufferLoad(int2(p) + int2(1,1), index), fract(p.x));
    return mix(a, b, fract(p.y));
}

fn hash44(p: float4) -> float4 {
	var p4 = fract(p * float4(.1031, .1030, .0973, .1099));
    p4 = p4 + dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
}

let dt = 1.;
let n = float2(0., 1.);
let e = float2(1., 0.);
let s = float2(0., -1.);
let w = float2(-1., 0.);

fn A(fragCoord: float2) -> float4 {
    let resolution = float2(textureDimensions(col));
    //return textureLoad(tex, int2(fragCoord), 0, 0);
    return bufferLoad(int2(fragCoord), 0);
}

fn B(fragCoord: float2) -> float4 {
    let resolution = float2(textureDimensions(col));
    //return textureSampleLevel(tex, bilinear, fract(fragCoord / resolution), 1, 0.);
    return bufferSample(fract(fragCoord / resolution), 1);
}

fn T(fragCoord: float2) -> float4 {
    return B(fragCoord - dt * B(fragCoord).xy);
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_velocity([[builtin(global_invocation_id)]] global_id: uint3) {
    let u = float2(global_id.xy) + 0.5;
    var r = T(u);
    r.x = r.x - dt * 0.25 * (T(u+e).z - T(u+w).z);
    r.y = r.y - dt * 0.25 * (T(u+n).z - T(u+s).z);

    if (params.frame < 3) { r = float4(0.); }
    //textureStore(texs, int2(global_id.xy), 0, r);
    bufferStore(int2(global_id.xy), 0, r);
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_pressure([[builtin(global_invocation_id)]] global_id: uint3) {
    let resolution = float2(textureDimensions(col));
    let u = float2(global_id.xy) + 0.5;
    var r = A(u);
    r.z = r.z - dt * 0.25 * (A(u+e).x - A(u+w).x + A(u+n).y - A(u+s).y);

    let t = float(params.frame) / 120.;
    let o = resolution/2. * (1. + .75 * float2(cos(t/15.), sin(2.7*t/15.)));
    r = mix(r, float4(0.5 * sin(dt * 2. * t) * sin(dt * t), 0., r.z, 1.), exp(-0.2 * length(u - o)));
    //textureStore(texs, int2(global_id.xy), 1, r);
    bufferStore(int2(global_id.xy), 1, r);
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_caustics([[builtin(global_invocation_id)]] global_id: uint3) {
    let resolution = float2(textureDimensions(col));
    for (var i = 0; i < 25; i = i+1) {
        let h = hash44(float4(float2(global_id.xy), float(params.frame), float(i)));
        var p = float2(global_id.xy) + h.xy;
        let z = mix(.3, 1., h.z);
        let c = max(cos(z*6.2+float4(1.,2.,3.,4.)),float4(0.));
        let n = A(p + float2(0., 1.));
        let e = A(p + float2(1., 0.));
        let s = A(p - float2(0., 1.));
        let w = A(p - float2(1., 0.));
        let grad = 0.25 * float2(e.z - w.z, n.z - s.z);
        p = p + 1e5 * grad * z;
        p = fract(p / resolution) * resolution;
        let id = int(p.x) + int(p.y) * int(resolution.x);
        atomicAdd(&buf.data[id*4+0], int(c.x * 256.));
        atomicAdd(&buf.data[id*4+1], int(c.y * 256.));
        atomicAdd(&buf.data[id*4+2], int(c.z * 256.));
    }
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_image([[builtin(global_invocation_id)]] global_id: uint3) {
    let resolution = float2(textureDimensions(col));
    let id = int(global_id.x) + int(global_id.y) * int(resolution.x);
    let x = float(atomicLoad(&buf.data[id*4+0]));
    let y = float(atomicLoad(&buf.data[id*4+1]));
    let z = float(atomicLoad(&buf.data[id*4+2]));
    var r = float3(x, y, z) / 256.;
    r = r * sqrt(r) / 5e3;
    r = r * float3(.5, .75, 1.);
    textureStore(col, int2(global_id.xy), float4(r, 1.));
    atomicStore(&buf.data[id*4+0], int(x * .9));
    atomicStore(&buf.data[id*4+1], int(y * .9));
    atomicStore(&buf.data[id*4+2], int(z * .9));
}
