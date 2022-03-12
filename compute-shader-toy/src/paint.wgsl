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
    values: array<atomic<u32>>;
};

[[group(0), binding(0)]] var<uniform> params: Params;
[[group(0), binding(1)]] var outputTex: texture_storage_2d<rgba8unorm,write>;
[[group(0), binding(2)]] var<storage,read_write> StorageBuffer0: StorageBuffer;
[[group(0), binding(3)]] var<storage,read_write> StorageBuffer1: StorageBuffer;

// https://www.shadertoy.com/view/lstGDs
// Created by inigo quilez - iq/2016
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

fn p2c(p: vec2<f32>) -> vec2<f32> {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    var q = p/resolution.xy;
    q = -1.0 + 2.0*q;
    q.x = q.x * resolution.x/resolution.y;
    return (q - vec2<f32>(0.5,0.0))*1.1;
}

fn c2p(c: vec2<f32>) -> vec2<f32> {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    var q = c/1.1 + vec2<f32>(0.5,0.0);
    q.x = q.x * resolution.y/resolution.x;
	q = (q+1.0)/2.0;
    return q*resolution.xy;
}

fn rand(seed: ptr<function, u32>) -> vec2<f32> {
    *seed = *seed*0x343fdu + 0x269ec3u; let x = *seed;
    *seed = *seed*0x343fdu + 0x269ec3u; let y = *seed;
    return vec2<f32>(f32((x>>16u)&32767u), f32((y>>16u)&32767u))/32767.0;
}

fn smoothstep(edge0: vec4<f32>, edge1: vec4<f32>, x: vec4<f32>) -> vec4<f32> {
    let t = clamp((x - edge0) / (edge1 - edge0), vec4<f32>(0.0), vec4<f32>(1.0));
    return t * t * (3.0 - 2.0 * t);
}

[[stage(compute), workgroup_size(16, 16)]]
fn main2([[builtin(global_invocation_id)]] global_ix: vec3<u32>) {
    let id = global_ix.x + global_ix.y * params.width;
    StorageBuffer0.values[id] = StorageBuffer0.values[id] * 9u / 10u;
}

[[stage(compute), workgroup_size(16, 16)]]
fn main([[builtin(global_invocation_id)]] global_ix: vec3<u32>) {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    let id = global_ix.x + global_ix.y * params.width;

    // Shadertoy-like code can go here.
    var seed = params.iFrame * params.width * params.height + id;
    seed = seed ^ (seed<<13u);

    for (var i = 0; i < 10; i = i+1) {
        var p = rand(&seed) * 2. - 1.;
        p = p + .2 * sin(15. * p.yx + 3. + params.iTime);
        p = p + .1 * sin(15. * length(p) + 2. + params.iTime);
        p = (p + 1.)/2. * resolution.xy;
        let id1 = u32(p.x) + u32(p.y) * params.width;
        atomicAdd(&StorageBuffer0.values[id1], 1u);
    }

    let z = f32(StorageBuffer0.values[id]);
    var f = vec4<f32>(z) / 500.;
    f = smoothstep(vec4<f32>(0.), vec4<f32>(1.), 2.5 * pow(f, vec4<f32>(1.5, 1.4, 1.3, 1.)));
    textureStore(outputTex, vec2<i32>(global_ix.xy), f);
}
