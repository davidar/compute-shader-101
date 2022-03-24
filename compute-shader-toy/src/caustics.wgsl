struct Params {
    width: u32;
    height: u32;
    frame: u32;
};

struct AtomicStorageBuffer {
    data: array<atomic<u32>>;
};

struct FloatStorageBuffer {
    data: array<vec4<f32>>;
};

[[group(0), binding(0)]] var<uniform> params: Params;
[[group(0), binding(1)]] var col: texture_storage_2d<rgba16float,write>;
[[group(0), binding(2)]] var<storage,read_write> buf: AtomicStorageBuffer;
//[[group(0), binding(3)]] var tex: texture_2d_array<f32>;
//[[group(0), binding(4)]] var texs: texture_storage_2d_array<rgba16float,write>;
//[[group(0), binding(5)]] var nearest: sampler;
//[[group(0), binding(6)]] var bilinear: sampler;
[[group(0), binding(7)]] var<storage,read_write> fbuf: FloatStorageBuffer;

fn bufferStore(pos: vec3<u32>, value: vec4<f32>) {
    fbuf.data[pos.x + pos.y * params.width + pos.z * params.width * params.height] = value;
}

fn bufferLoad(pos: vec3<u32>) -> vec4<f32> {
    return fbuf.data[pos.x + pos.y * params.width + pos.z * params.width * params.height];
}

fn bufferSample(uv: vec2<f32>, index: u32) -> vec4<f32> {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    let p = clamp(uv, vec2<f32>(0.), vec2<f32>(1.)) * resolution - 0.5;
    let pos = vec3<u32>(vec2<u32>(floor(p)), index);
    let a = mix(bufferLoad(pos + vec3<u32>(0u,0u,0u)), bufferLoad(pos + vec3<u32>(1u,0u,0u)), fract(p.x));
    let b = mix(bufferLoad(pos + vec3<u32>(0u,1u,0u)), bufferLoad(pos + vec3<u32>(1u,1u,0u)), fract(p.x));
    return mix(a, b, fract(p.y));
}

fn hash44(p: vec4<f32>) -> vec4<f32> {
	var p4 = fract(p * vec4<f32>(.1031, .1030, .0973, .1099));
    p4 = p4 + dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
}

let dt = 1.;
let n = vec2<f32>(0., 1.);
let e = vec2<f32>(1., 0.);
let s = vec2<f32>(0., -1.);
let w = vec2<f32>(-1., 0.);

fn A(fragCoord: vec2<f32>) -> vec4<f32> {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    //return textureLoad(tex, vec2<i32>(fragCoord), 0, 0);
    return bufferLoad(vec3<u32>(vec2<u32>(fragCoord), 0u));
}

fn B(fragCoord: vec2<f32>) -> vec4<f32> {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    //return textureSampleLevel(tex, bilinear, fract(fragCoord / resolution), 1, 0.);
    return bufferSample(fract(fragCoord / resolution), 1u);
}

fn T(fragCoord: vec2<f32>) -> vec4<f32> {
    return B(fragCoord - dt * B(fragCoord).xy);
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_velocity([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let u = vec2<f32>(global_id.xy) + 0.5;
    var r = T(u);
    r.x = r.x - dt * 0.25 * (T(u+e).z - T(u+w).z);
    r.y = r.y - dt * 0.25 * (T(u+n).z - T(u+s).z);

    if (params.frame < 3u) { r = vec4<f32>(0.); }
    //textureStore(texs, vec2<i32>(global_id.xy), 0, r);
    bufferStore(vec3<u32>(global_id.xy, 0u), r);
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_pressure([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    let u = vec2<f32>(global_id.xy) + 0.5;
    var r = A(u);
    r.z = r.z - dt * 0.25 * (A(u+e).x - A(u+w).x + A(u+n).y - A(u+s).y);

    let t = f32(params.frame) / 120.;
    let o = resolution/2. * (1. + .75 * vec2<f32>(cos(t/15.), sin(2.7*t/15.)));
    r = mix(r, vec4<f32>(0.5 * sin(dt * 2. * t) * sin(dt * t), 0., r.z, 1.), exp(-0.2 * length(u - o)));
    //textureStore(texs, vec2<i32>(global_id.xy), 1, r);
    bufferStore(vec3<u32>(global_id.xy, 1u), r);
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_caustics([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    for (var i = 0; i < 25; i = i+1) {
        let h = hash44(vec4<f32>(vec2<f32>(global_id.xy), f32(params.frame), f32(i)));
        var p = vec2<f32>(global_id.xy) + h.xy;
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
        atomicAdd(&buf.data[id*4u+0u], u32(c.x * 256.));
        atomicAdd(&buf.data[id*4u+1u], u32(c.y * 256.));
        atomicAdd(&buf.data[id*4u+2u], u32(c.z * 256.));
    }
}

[[stage(compute), workgroup_size(16, 16)]]
fn main_image([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let id = global_id.x + global_id.y * params.width;
    let x = f32(atomicLoad(&buf.data[id*4u+0u]));
    let y = f32(atomicLoad(&buf.data[id*4u+1u]));
    let z = f32(atomicLoad(&buf.data[id*4u+2u]));
    var r = vec3<f32>(x, y, z) / 256.;
    r = r * sqrt(r) / 5e3;
    r = r * vec3<f32>(.5, .75, 1.);
    textureStore(col, vec2<i32>(global_id.xy), vec4<f32>(r, 1.));
    atomicStore(&buf.data[id*4u+0u], u32(x * .9));
    atomicStore(&buf.data[id*4u+1u], u32(y * .9));
    atomicStore(&buf.data[id*4u+2u], u32(z * .9));
}
