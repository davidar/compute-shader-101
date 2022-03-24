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

use wgpu::{BufferUsages, Extent3d};

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

#[cfg(not(target_arch = "wasm32"))]
pub struct Spawner<'a> {
    executor: async_executor::LocalExecutor<'a>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> Spawner<'a> {
    fn new() -> Self {
        Self {
            executor: async_executor::LocalExecutor::new(),
        }
    }

    #[allow(dead_code)]
    pub fn spawn_local(&self, future: impl std::future::Future<Output = ()> + 'a) {
        self.executor.spawn(future).detach();
    }

    fn run_until_stalled(&self) {
        while self.executor.try_tick() {}
    }
}

#[cfg(target_arch = "wasm32")]
pub struct Spawner {}

#[cfg(target_arch = "wasm32")]
impl Spawner {
    fn new() -> Self {
        Self {}
    }

    #[allow(dead_code)]
    pub fn spawn_local(&self, future: impl std::future::Future<Output = ()> + 'static) {
        wasm_bindgen_futures::spawn_local(future);
    }
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let shader = include_str!("caustics.wgsl");
    let entry_points = ["main_velocity", "main_pressure", "main_caustics", "main_image"];
    //let shader = include_str!("buddhabrot.wgsl");
    //let entry_points = ["main_hist", "main_image"];

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
    window.set_inner_size(winit::dpi::PhysicalSize::new(1280, 720));
    let size = window.inner_size();
    let format = surface.get_preferred_format(&adapter).unwrap();
    surface.configure(&device, &wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: format,
        width: size.width,
        height: size.height,
        //present_mode: wgpu::PresentMode::Fifo, // vsync
        present_mode: wgpu::PresentMode::Mailbox,
    });

    // uniforms
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 3 * 4,
        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    let img = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
    });
    /*let buf_read = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 4,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
    });
    let buf_write = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 4,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
    });*/
    let sb0 = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (4 * 4 * size.width * size.height).into(),
        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    let sbf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (4 * 4 * 4 * size.width * size.height).into(),
        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    // compute pipeline
    let compute_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: false
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            /*wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },*/
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: false
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&compute_bind_group_layout],
        push_constant_ranges: &[],
    });
    let compute_pipelines = entry_points.map(|name| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: name,
        }));
    let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &compute_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&img.create_view(&Default::default())) },
            wgpu::BindGroupEntry { binding: 2, resource: sb0.as_entire_binding() },
            /*wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&buf_read.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&buf_write.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&device.create_sampler(&Default::default())) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            })) },*/
            wgpu::BindGroupEntry { binding: 7, resource: sbf.as_entire_binding() },
        ],
    });

    // render pipeline
    let render_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
    });
    let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&render_bind_group_layout],
            push_constant_ranges: &[],
        })),
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
    let render_pipeline_srgb = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&render_bind_group_layout],
            push_constant_ranges: &[],
        })),
        vertex: wgpu::VertexState {
            module: &render_shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &render_shader,
            entry_point: "fs_main_srgb",
            targets: &[format.into()],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });
    let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &render_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&img.create_view(&Default::default())) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&device.create_sampler(&Default::default())) },
        ],
    });

    let spawner = Spawner::new();
    let mut staging_belt = wgpu::util::StagingBelt::new(0x100);
    let mut frame_count: u32 = 0;
    #[cfg(not(target_arch = "wasm32"))]
    let mut last_frame_inst = std::time::Instant::now();
    #[cfg(not(target_arch = "wasm32"))]
    let mut accum_time = 0.;
    let mut frame_count2 = 0;
    event_loop.run(move |event, _, control_flow| {
        // TODO: this may be excessive polling. It really should be synchronized with
        // swapchain presentation, but that's currently underbaked in wgpu.
        *control_flow = ControlFlow::Poll;
        match event {
            Event::RedrawRequested(_) => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    accum_time += last_frame_inst.elapsed().as_secs_f32();
                    last_frame_inst = std::time::Instant::now();
                    frame_count2 += 1;
                    if frame_count2 == 100 {
                        println!(
                            "Avg frame time {}ms",
                            accum_time * 1000.0 / frame_count2 as f32
                        );
                        accum_time = 0.0;
                        frame_count2 = 0;
                    }
                }
                let frame = surface
                    .get_current_texture()
                    .expect("error getting texture from swap chain");
                let params_data = [size.width, size.height, frame_count];
                let params_bytes = bytemuck::bytes_of(&params_data);
                let mut encoder = device.create_command_encoder(&Default::default());
                staging_belt.write_buffer(&mut encoder, &params, 0, wgpu::BufferSize::new(params_bytes.len() as wgpu::BufferAddress).unwrap(), &device).copy_from_slice(params_bytes);
                staging_belt.finish();
                let run_compute_pass = |encoder: &mut wgpu::CommandEncoder, pipeline| {
                    {
                        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
                        compute_pass.set_pipeline(pipeline);
                        compute_pass.set_bind_group(0, &compute_bind_group, &[]);
                        compute_pass.dispatch(size.width / 16, size.height / 16, 1);
                    }
                    /*encoder.copy_texture_to_texture(
                        wgpu::ImageCopyTexture {
                            texture: &buf_write,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::ImageCopyTexture {
                            texture: &buf_read,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        Extent3d {
                            width: size.width,
                            height: size.height,
                            depth_or_array_layers: 4,
                        });*/
                };
                for _ in 0..1 {
                    for pipeline in &compute_pipelines {
                        run_compute_pass(&mut encoder, pipeline);
                    }
                    frame_count += 1;
                }
                // blit the output texture to the framebuffer
                {
                    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                        format: Some(format),
                        ..Default::default()
                    });
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
                    match format {
                        // TODO use sRGB viewFormats instead once the API stabilises?
                        wgpu::TextureFormat::Bgra8Unorm => render_pass.set_pipeline(&render_pipeline_srgb),
                        wgpu::TextureFormat::Bgra8UnormSrgb => render_pass.set_pipeline(&render_pipeline),
                        _ => panic!("unrecognised surface format")
                    }
                    render_pass.set_bind_group(0, &render_bind_group, &[]);
                    render_pass.draw(0..3, 0..1);
                }
                queue.submit(Some(encoder.finish()));
                spawner.spawn_local(staging_belt.recall());
                frame.present();
                #[cfg(not(target_arch = "wasm32"))]
                spawner.run_until_stalled();
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

// https://github.com/rust-windowing/winit/blob/master/examples/web.rs
// https://github.com/gfx-rs/wgpu/blob/master/wgpu/examples/framework.rs

fn main() {
    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init_from_env(
            env_logger::Env::default().filter_or(env_logger::DEFAULT_FILTER_ENV, "info"));
        pollster::block_on(run(event_loop, window));
    }

    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        let log_list = wasm::create_log_list(&window);

        use wasm_bindgen::{prelude::*, JsCast};

        wasm_bindgen_futures::spawn_local(async move {
            let setup = run(event_loop, window).await;
            let start_closure = Closure::once_into_js(move || setup);

            // make sure to handle JS exceptions thrown inside start.
            // Otherwise wasm_bindgen_futures Queue would break and never handle any tasks again.
            // This is required, because winit uses JS exception for control flow to escape from `run`.
            if let Err(error) = call_catch(&start_closure) {
                let is_control_flow_exception = error.dyn_ref::<js_sys::Error>().map_or(false, |e| {
                    e.message().includes("Using exceptions for control flow", 0)
                });

                if !is_control_flow_exception {
                    web_sys::console::error_1(&error);
                }
            }

            #[wasm_bindgen]
            extern "C" {
                #[wasm_bindgen(catch, js_namespace = Function, js_name = "prototype.call.call")]
                fn call_catch(this: &JsValue) -> Result<(), JsValue>;
            }
        });
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::*;
    use winit::{event::Event, window::Window};

    #[wasm_bindgen(start)]
    pub fn run() {
        console_log::init_with_level(log::Level::Debug).expect("error initializing logger");

        super::main();
    }

    pub fn create_log_list(window: &Window) -> web_sys::Element {
        use winit::platform::web::WindowExtWebSys;

        let canvas = window.canvas();

        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();
        let body = document.body().unwrap();

        // Set a background color for the canvas to make it easier to tell the where the canvas is for debugging purposes.
        canvas.style().set_css_text("background-color: crimson;");
        body.append_child(&canvas).unwrap();

        let log_header = document.create_element("h2").unwrap();
        log_header.set_text_content(Some("Event Log"));
        body.append_child(&log_header).unwrap();

        let log_list = document.create_element("ul").unwrap();
        body.append_child(&log_list).unwrap();
        log_list
    }

    pub fn log_event(log_list: &web_sys::Element, event: &Event<()>) {
        log::debug!("{:?}", event);

        // Getting access to browser logs requires a lot of setup on mobile devices.
        // So we implement this basic logging system into the page to give developers an easy alternative.
        // As a bonus its also kind of handy on desktop.
        if let Event::WindowEvent { event, .. } = &event {
            let window = web_sys::window().unwrap();
            let document = window.document().unwrap();
            let log = document.create_element("li").unwrap();
            log.set_text_content(Some(&format!("{:?}", event)));
            log_list
                .insert_before(&log, log_list.first_child().as_ref())
                .unwrap();
        }
    }
}
