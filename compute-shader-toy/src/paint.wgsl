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

struct U32s {
    values: array<atomic<u32>>;
};

[[group(0), binding(0)]] var<uniform> params: Params;
[[group(0), binding(1)]] var outputTex: texture_storage_2d<rgba8unorm,write>;

// A storage buffer, for reading and writing
[[group(0), binding(2)]] var<storage,read_write> pbuf: U32s;

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

[[stage(compute), workgroup_size(16, 16)]]
fn main([[builtin(global_invocation_id)]] global_ix: vec3<u32>) {
    let resolution = vec2<f32>(f32(params.width), f32(params.height));
    let fragCoord = vec2<f32>(global_ix.xy) / resolution - vec2<f32>(0.5, 0.5);
    let id = global_ix.x + global_ix.y * params.width;

    // Shadertoy-like code can go here.
    var seed = params.iFrame * params.width * params.height + id;
    seed = seed ^ (seed<<13u);

    // do 8 passes per frame (since texture memory bandwidh is slow, the
    // more computations I do per frame the better)
    for (var j = 0; j < 8; j = j+1) {
        // pick a random point, but not in H1 or H2
        var c = vec2<f32>(0.);
        for (var i = 0; i < 32; i = i+1) {
            let of = rand(&seed);
            c = p2c(of * resolution.xy) * 1.1;
            // test H1 and H2
            let c2 = dot(c,c);
            let d = c + vec2<f32>(1.,0.);
            let h1 = 256.0*c2*c2 - 96.0*c2 + 32.0*c.x - 3.0;
            let h2 = 16.0*dot(d,d) - 1.0;
            if (h1 > 0. && h2 > 0.) { break; }
        }

        // compute and draw orbit
        var z = vec2<f32>(0.);
        for (var i = 0; i < 512; i = i+1) {
            z = vec2<f32>(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
            if (dot(z,z) > 9.) { break; }
            if (i > 128) {
                let p1 = c2p(z);
                let p2 = c2p(vec2<f32>(z.x,-z.y));
                atomicAdd(&pbuf.values[u32(p1.x) + u32(p1.y) * params.width], 1u);
                atomicAdd(&pbuf.values[u32(p2.x) + u32(p2.y) * params.width], 1u);
            }
        }
    }

    let fragColor = vec4<f32>(1.) * f32(pbuf.values[id]) / 5e5 / params.iTime;
    textureStore(outputTex, vec2<i32>(global_ix.xy), fragColor);
}
