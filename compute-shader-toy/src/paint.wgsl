// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

struct Params {
    width: u32;
    height: u32;
    iFrame: u32;
    iTime: f32;
};

struct StorageBuffer {
    data: array<atomic<u32>>;
};

[[group(0), binding(0)]] var<uniform> params: Params;
[[group(0), binding(1)]] var outputTex: texture_storage_2d<rgba16float,write>;
[[group(0), binding(2)]] var<storage,read_write> storageBuffer: StorageBuffer;
[[group(0), binding(4)]] var inputTex: texture_2d<f32>;
[[group(0), binding(5)]] var inputSampler: sampler;

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

fn A(fragCoord: vec2<f32>) -> vec4<f32> {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    return textureSampleLevel(inputTex, inputSampler, fract(fragCoord / resolution), 0.);
}

fn T(fragCoord: vec2<f32>) -> vec4<f32> {
    return A(fragCoord - dt * A(fragCoord).xy);
}

[[stage(compute), workgroup_size(16, 16)]]
fn bufferA([[builtin(global_invocation_id)]] global_ix: vec3<u32>) {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    let fragCoord = vec2<f32>(global_ix.xy) + 0.5;
    var r = T(fragCoord);
    let n = T(fragCoord + vec2<f32>(0., 1.));
    let e = T(fragCoord + vec2<f32>(1., 0.));
    let s = T(fragCoord - vec2<f32>(0., 1.));
    let w = T(fragCoord - vec2<f32>(1., 0.));
    r.x = r.x - dt * 0.25 * (e.z - w.z);
    r.y = r.y - dt * 0.25 * (n.z - s.z);

    if (params.iFrame < 3u) { r = vec4<f32>(0.); }
    textureStore(outputTex, vec2<i32>(global_ix.xy), r);
}

[[stage(compute), workgroup_size(16, 16)]]
fn bufferB([[builtin(global_invocation_id)]] global_ix: vec3<u32>) {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    let fragCoord = vec2<f32>(global_ix.xy) + 0.5;
    var r = A(fragCoord);
    let n = A(fragCoord + vec2<f32>(0., 1.));
    let e = A(fragCoord + vec2<f32>(1., 0.));
    let s = A(fragCoord - vec2<f32>(0., 1.));
    let w = A(fragCoord - vec2<f32>(1., 0.));
    r.z = r.z - dt * 0.25 * (e.x - w.x + n.y - s.y);

    let t = f32(params.iFrame) / 120.;
    let o = resolution/2. * (1. + .75 * vec2<f32>(cos(t/15.), sin(2.7*t/15.)));
    r = mix(r, vec4<f32>(0.5 * sin(dt * 2. * t) * sin(dt * t), 0., r.z, 1.), exp(-0.2 * length(fragCoord - o)));
    textureStore(outputTex, vec2<i32>(global_ix.xy), r);
}

[[stage(compute), workgroup_size(16, 16)]]
fn main([[builtin(global_invocation_id)]] global_ix: vec3<u32>) {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    for (var i = 0; i < 25; i = i+1) {
        let h = hash44(vec4<f32>(vec2<f32>(global_ix.xy), f32(params.iFrame), f32(i)));
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
    var f = vec4<f32>(0.);
    f.x = f32(storageBuffer.data[id*4u+0u]) / 256.;
    f.y = f32(storageBuffer.data[id*4u+1u]) / 256.;
    f.z = f32(storageBuffer.data[id*4u+2u]) / 256.;
    f = f * sqrt(f) / 5e3;
    f = f * vec4<f32>(.5, .75, 1., 1.);
    textureStore(outputTex, vec2<i32>(global_ix.xy), f);
    storageBuffer.data[id*4u+0u] = storageBuffer.data[id*4u+0u] * 9u / 10u;
    storageBuffer.data[id*4u+1u] = storageBuffer.data[id*4u+1u] * 9u / 10u;
    storageBuffer.data[id*4u+2u] = storageBuffer.data[id*4u+2u] * 9u / 10u;
}
