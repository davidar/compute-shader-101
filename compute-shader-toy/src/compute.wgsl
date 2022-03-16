struct Params {
    width: u32;
    height: u32;
    frame: u32;
};

struct StorageBuffer {
    data: array<atomic<u32>>;
};

[[group(0), binding(0)]] var<uniform> params: Params;
[[group(0), binding(1)]] var outputTex: texture_storage_2d<rgba16float,write>;
[[group(0), binding(2)]] var<storage,read_write> storageBuffer: StorageBuffer;
[[group(0), binding(3)]] var inputTexArr: texture_2d_array<f32>;
[[group(0), binding(4)]] var outputTexArr: texture_storage_2d_array<rgba16float,write>;
[[group(0), binding(5)]] var samplerLinear: sampler;

fn hash44(p: vec4<f32>) -> vec4<f32> {
	var p4 = fract(p * vec4<f32>(.1031, .1030, .0973, .1099));
    p4 = p4 + dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
}

fn smoothstep(edge0: vec4<f32>, edge1: vec4<f32>, x: vec4<f32>) -> vec4<f32> {
    let t = clamp((x - edge0) / (edge1 - edge0), vec4<f32>(0.0), vec4<f32>(1.0));
    return t * t * (3.0 - 2.0 * t);
}

let dt = 1.;
let n = vec2<f32>(0., 1.);
let e = vec2<f32>(1., 0.);
let s = vec2<f32>(0., -1.);
let w = vec2<f32>(-1., 0.);

fn A(fragCoord: vec2<f32>) -> vec4<f32> {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    return textureSampleLevel(inputTexArr, samplerLinear, fract(fragCoord / resolution), 0, 0.);
}

fn B(fragCoord: vec2<f32>) -> vec4<f32> {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    return textureSampleLevel(inputTexArr, samplerLinear, fract(fragCoord / resolution), 1, 0.);
}

fn T(fragCoord: vec2<f32>) -> vec4<f32> {
    return B(fragCoord - dt * B(fragCoord).xy);
}

[[stage(compute), workgroup_size(16, 16)]]
fn bufferA([[builtin(global_invocation_id)]] global_ix: vec3<u32>) {
    let u = vec2<f32>(global_ix.xy) + 0.5;
    var r = T(u);
    r.x = r.x - dt * 0.25 * (T(u+e).z - T(u+w).z);
    r.y = r.y - dt * 0.25 * (T(u+n).z - T(u+s).z);

    if (params.frame < 3u) { r = vec4<f32>(0.); }
    textureStore(outputTexArr, vec2<i32>(global_ix.xy), 0, r);
}

[[stage(compute), workgroup_size(16, 16)]]
fn bufferB([[builtin(global_invocation_id)]] global_ix: vec3<u32>) {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    let u = vec2<f32>(global_ix.xy) + 0.5;
    var r = A(u);
    r.z = r.z - dt * 0.25 * (A(u+e).x - A(u+w).x + A(u+n).y - A(u+s).y);

    let t = f32(params.frame) / 120.;
    let o = resolution/2. * (1. + .75 * vec2<f32>(cos(t/15.), sin(2.7*t/15.)));
    r = mix(r, vec4<f32>(0.5 * sin(dt * 2. * t) * sin(dt * t), 0., r.z, 1.), exp(-0.2 * length(u - o)));
    textureStore(outputTexArr, vec2<i32>(global_ix.xy), 1, r);
}

[[stage(compute), workgroup_size(16, 16)]]
fn main([[builtin(global_invocation_id)]] global_ix: vec3<u32>) {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    for (var i = 0; i < 25; i = i+1) {
        let h = hash44(vec4<f32>(vec2<f32>(global_ix.xy), f32(params.frame), f32(i)));
        var p = vec2<f32>(global_ix.xy) + h.xy;
        let z = mix(.3, 1., h.z);
        let c = max(cos(z*6.2+vec4<f32>(1.,2.,3.,4.)),vec4<f32>(0.));
        let n = A(p + vec2<f32>(0., 1.));
        let e = A(p + vec2<f32>(1., 0.));
        let s = A(p - vec2<f32>(0., 1.));
        let w = A(p - vec2<f32>(1., 0.));
        let grad = 0.25 * vec2<f32>(e.z - w.z, n.z - s.z);
        p = p + 1e5 * grad * z;
        p = fract(p / resolution) * resolution;
        let id = u32(p.x) + u32(p.y) * params.width;
        atomicAdd(&storageBuffer.data[id*4u+0u], u32(c.x * 256.));
        atomicAdd(&storageBuffer.data[id*4u+1u], u32(c.y * 256.));
        atomicAdd(&storageBuffer.data[id*4u+2u], u32(c.z * 256.));
    }
}

[[stage(compute), workgroup_size(16, 16)]]
fn mainImage([[builtin(global_invocation_id)]] global_ix: vec3<u32>) {
    let id = global_ix.x + global_ix.y * params.width;
    let x = f32(atomicLoad(&storageBuffer.data[id*4u+0u]));
    let y = f32(atomicLoad(&storageBuffer.data[id*4u+1u]));
    let z = f32(atomicLoad(&storageBuffer.data[id*4u+2u]));
    var r = vec3<f32>(x, y, z) / 256.;
    r = r * sqrt(r) / 5e3;
    r = r * vec3<f32>(.5, .75, 1.);
    textureStore(outputTex, vec2<i32>(global_ix.xy), vec4<f32>(r, 1.));
    atomicStore(&storageBuffer.data[id*4u+0u], u32(x * .9));
    atomicStore(&storageBuffer.data[id*4u+1u], u32(y * .9));
    atomicStore(&storageBuffer.data[id*4u+2u], u32(z * .9));
}