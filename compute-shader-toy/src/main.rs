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

//! A simple compute shader example that draws into a window, based on wgpu.

use wgpu::util::DeviceExt;
use wgpu::{BufferUsages, Extent3d};

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: Default::default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .expect("error finding adapter");

    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .await
        .expect("error creating device");
    let size = window.inner_size();
    let format = surface.get_preferred_format(&adapter).unwrap();
    surface.configure(&device, &wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    });

    // uniforms
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 * 4,
        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    let texture_descriptor = wgpu::TextureDescriptor {
        label: None,
        size: Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
    };
    let img = device.create_texture(&texture_descriptor);
    let sb0 = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (4 * size.width * size.height).into(),
        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    // compute pipeline
    let compute_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("paint.wgsl").into()),
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &compute_shader,
        entry_point: "main",
    });
    let compute_pipeline2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &compute_shader,
        entry_point: "main2",
    });
    let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&img.create_view(&Default::default())),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: sb0.as_entire_binding(),
            },
        ],
    });
    let compute_bind_group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &compute_pipeline2.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: sb0.as_entire_binding(),
            },
        ],
    });

    // render pipeline
    let render_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("copy.wgsl").into()),
    });
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &render_shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &render_shader,
            entry_point: "fs_main",
            targets: &[format.into()],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });
    let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &render_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&img.create_view(&Default::default())),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&device.create_sampler(&Default::default())),
            },
        ],
    });

    let start_time = std::time::Instant::now();
    let mut frame_count: u32 = 0;
    event_loop.run(move |event, _, control_flow| {
        // TODO: this may be excessive polling. It really should be synchronized with
        // swapchain presentation, but that's currently underbaked in wgpu.
        *control_flow = ControlFlow::Poll;
        match event {
            Event::RedrawRequested(_) => {
                let frame = surface
                    .get_current_texture()
                    .expect("error getting texture from swap chain");
                let time: f32 = start_time.elapsed().as_micros() as f32 * 1e-6;
                let params_data = [size.width, size.height, frame_count, time.to_bits()];
                let params_bytes = bytemuck::bytes_of(&params_data);
                let params_host = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: params_bytes,
                    usage: BufferUsages::COPY_SRC,
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(&params_host, 0, &params, 0, params_bytes.len().try_into().unwrap());
                {
                    let mut compute_pass = encoder.begin_compute_pass(&Default::default());
                    compute_pass.set_pipeline(&compute_pipeline2);
                    compute_pass.set_bind_group(0, &compute_bind_group2, &[]);
                    compute_pass.dispatch(size.width / 16, size.height / 16, 1);
                }
                {
                    let mut compute_pass = encoder.begin_compute_pass(&Default::default());
                    compute_pass.set_pipeline(&compute_pipeline);
                    compute_pass.set_bind_group(0, &compute_bind_group, &[]);
                    compute_pass.dispatch(size.width / 16, size.height / 16, 1);
                }
                // We use a render pipeline just to copy the output buffer of the compute shader to the
                // swapchain. It would be nice if we could skip this, but swapchains with storage usage
                // are not fully portable.
                {
                    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                store: true,
                            },
                        }],
                        depth_stencil_attachment: None,
                    });
                    render_pass.set_pipeline(&render_pipeline);
                    render_pass.set_bind_group(0, &render_bind_group, &[]);
                    render_pass.draw(0..3, 0..2);
                }
                queue.submit(Some(encoder.finish()));
                frame.present();
                frame_count += 1;
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => (),
        }
    });
}

fn main() {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();
    pollster::block_on(run(event_loop, window));
}
