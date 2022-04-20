use crevice::std140::AsStd140;
use futures::FutureExt;
use glam::Mat4;
use kiss_engine_wgpu::{
    BindingResource, Device, RenderPipelineDesc, ShaderSettings, VertexBufferLayout,
};
use wasm_webxr_helpers::{button_click_future, create_button};
use wgpu::util::DeviceExt;

mod assets;

use assets::{
    load_gltf, load_single_pixel_image, prune_fetched_images, FetchedImages, ModelLoadContext,
};

enum AnisotrophicFilteringLevel {
    L2 = 2,
    L4 = 4,
    L8 = 8,
    L16 = 16,
}

struct PerformanceSettings {
    anisotrophic_filtering_level: Option<AnisotrophicFilteringLevel>,
}

async fn run() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    let level: log::Level = wasm_web_helpers::parse_url_query_string_from_window("RUST_LOG")
        .and_then(|x| x.parse().ok())
        .unwrap_or(log::Level::Info);
    console_log::init_with_level(level).expect("could not initialize logger");

    let vr_button = create_button("Start VR");
    let ar_button = create_button("Start AR");
    let inline_button = create_button("Start inline rendering");

    let start_vr_future = button_click_future(&vr_button);
    let start_ar_future = button_click_future(&ar_button);
    let start_inline_future = button_click_future(&inline_button);

    let canvas = wasm_webxr_helpers::Canvas::default();
    let webgl2_context = canvas.create_webgl2_context();

    let navigator = web_sys::window().unwrap().navigator();
    let xr = navigator.xr();

    let mode = futures::select! {
        _ = Box::pin(start_vr_future.fuse()) => web_sys::XrSessionMode::ImmersiveVr,
        _ = Box::pin(start_ar_future.fuse()) => web_sys::XrSessionMode::ImmersiveAr,
        _ = Box::pin(start_inline_future.fuse()) => web_sys::XrSessionMode::Inline,
    };

    let reference_space_type = match mode {
        web_sys::XrSessionMode::Inline => web_sys::XrReferenceSpaceType::Viewer,
        _ => web_sys::XrReferenceSpaceType::Local,
    };

    let xr_session: web_sys::XrSession =
        wasm_bindgen_futures::JsFuture::from(xr.request_session(mode))
            .await
            .unwrap()
            .into();

    let xr_gl_layer =
        web_sys::XrWebGlLayer::new_with_web_gl2_rendering_context(&xr_session, &webgl2_context)
            .unwrap();

    let framebuffer_height = xr_gl_layer.framebuffer_height();
    let framebuffer_width = xr_gl_layer.framebuffer_width();

    let mut render_state_init = web_sys::XrRenderStateInit::new();
    render_state_init
        .depth_near(0.001)
        .base_layer(Some(&xr_gl_layer));
    xr_session.update_render_state_with_state(&render_state_init);

    let reference_space: web_sys::XrReferenceSpace = wasm_bindgen_futures::JsFuture::from(
        xr_session.request_reference_space(reference_space_type),
    )
    .await
    .unwrap()
    .into();

    let backend = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);
    let instance = wgpu::Instance::new(backend);
    let surface = unsafe { instance.create_surface(&canvas) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .expect("No suitable GPU adapters found on the system!");

    let adapter_info = adapter.get_info();
    log::info!(
        "Using {} with the {:?} backend",
        adapter_info.name,
        adapter_info.backend
    );

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device"),
                features: Default::default(),
                limits: wgpu::Limits {
                    max_texture_dimension_1d: framebuffer_width.max(framebuffer_height).max(2048),
                    max_texture_dimension_2d: framebuffer_width.max(framebuffer_height).max(2048),
                    ..wgpu::Limits::downlevel_webgl2_defaults()
                },
            },
            None,
        )
        .await
        .expect("Unable to find a suitable GPU adapter!");

    let mut device = Device::new(device);

    let mut fetched_images = FetchedImages::default();

    let base_url = url::Url::parse("http://localhost:8000").unwrap();

    let performance_settings = PerformanceSettings {
        anisotrophic_filtering_level: Some(AnisotrophicFilteringLevel::L16),
    };

    let linear_sampler = device.create_resource(
        device.inner.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: performance_settings
                .anisotrophic_filtering_level
                .map(|level| std::num::NonZeroU8::new(level as u8).unwrap()),
            ..Default::default()
        }),
    );

    let pbr_pipeline = kiss_engine_wgpu::create_render_pipeline(
        &device.inner,
        "pbr pipeline",
        device.get_shader(
            "vertex.spv",
            include_bytes!("../vertex.spv"),
            ShaderSettings {
                entry_point: "vertex",
                ..Default::default()
            },
        ),
        device.get_shader(
            "fragment.spv",
            include_bytes!("../fragment.spv"),
            ShaderSettings {
                entry_point: "fragment",
                ..Default::default()
            },
        ),
        RenderPipelineDesc {
            primitive: wgpu::PrimitiveState {
                // as we're flipping things in the shaders.
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_compare: wgpu::CompareFunction::Less,
            ..Default::default()
        },
        &[
            VertexBufferLayout {
                location: 0,
                format: wgpu::VertexFormat::Float32x3,
                step_mode: wgpu::VertexStepMode::Vertex,
            },
            VertexBufferLayout {
                location: 1,
                format: wgpu::VertexFormat::Float32x3,
                step_mode: wgpu::VertexStepMode::Vertex,
            },
            VertexBufferLayout {
                location: 2,
                format: wgpu::VertexFormat::Float32x2,
                step_mode: wgpu::VertexStepMode::Vertex,
            },
            VertexBufferLayout {
                location: 3,
                format: wgpu::VertexFormat::Float32,
                step_mode: wgpu::VertexStepMode::Instance,
            },
        ],
        &[wgpu::TextureFormat::Rgba8Unorm],
        Some(wgpu::TextureFormat::Depth32Float),
    );

    let pbr_alpha_clipped_pipeline = kiss_engine_wgpu::create_render_pipeline(
        &device.inner,
        "pbr alpha clipped pipeline",
        device.get_shader(
            "vertex.spv",
            include_bytes!("../vertex.spv"),
            ShaderSettings {
                entry_point: "vertex",
                ..Default::default()
            },
        ),
        device.get_shader(
            "fragment_alpha_clipped.spv",
            include_bytes!("../fragment_alpha_clipped.spv"),
            ShaderSettings {
                entry_point: "fragment_alpha_clipped",
                ..Default::default()
            },
        ),
        RenderPipelineDesc {
            primitive: wgpu::PrimitiveState {
                // as we're flipping things in the shaders.
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_compare: wgpu::CompareFunction::Less,
            ..Default::default()
        },
        &[
            VertexBufferLayout {
                location: 0,
                format: wgpu::VertexFormat::Float32x3,
                step_mode: wgpu::VertexStepMode::Vertex,
            },
            VertexBufferLayout {
                location: 1,
                format: wgpu::VertexFormat::Float32x3,
                step_mode: wgpu::VertexStepMode::Vertex,
            },
            VertexBufferLayout {
                location: 2,
                format: wgpu::VertexFormat::Float32x2,
                step_mode: wgpu::VertexStepMode::Vertex,
            },
            VertexBufferLayout {
                location: 3,
                format: wgpu::VertexFormat::Float32,
                step_mode: wgpu::VertexStepMode::Instance,
            },
        ],
        &[wgpu::TextureFormat::Rgba8Unorm],
        Some(wgpu::TextureFormat::Depth32Float),
    );

    let mut context = ModelLoadContext {
        url: url::Url::options()
            .base_url(Some(&base_url))
            .parse("sample_models/Sponza/glTF/Sponza.gltf")
            .unwrap(),
        device: &device,
        queue: &queue,
        fetched_images: &mut fetched_images,
        pbr_pipeline: &pbr_pipeline,
        pbr_alpha_clipped_pipeline: &pbr_alpha_clipped_pipeline,
        black_image: load_single_pixel_image(
            &device,
            &queue,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &[0, 0, 0, 255],
        ),
        default_metallic_roughness_image: load_single_pixel_image(
            &device,
            &queue,
            wgpu::TextureFormat::Rgba8Unorm,
            &[0, 255, 0, 255],
        ),
        flat_normals_image: load_single_pixel_image(
            &device,
            &queue,
            wgpu::TextureFormat::Rgba8Unorm,
            &[0, 255, 0, 255],
        ),
        white_image: load_single_pixel_image(
            &device,
            &queue,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &[255, 255, 255, 255],
        ),
    };
    let model = load_gltf(&mut context).await;

    let left_eye_uniform_buffer =
        device.create_resource(device.inner.create_buffer(&wgpu::BufferDescriptor {
            label: Some("left eye uniform buffer"),
            size: std::mem::size_of::<shared_structs::Uniforms>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        }));

    let right_eye_uniform_buffer =
        device.create_resource(device.inner.create_buffer(&wgpu::BufferDescriptor {
            label: Some("right eye uniform buffer"),
            size: std::mem::size_of::<shared_structs::Uniforms>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        }));

    let instance_buffer = device.create_resource(device.inner.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("right eye uniform buffer"),
            contents: bytemuck::bytes_of(&0.25_f32),
            usage: wgpu::BufferUsages::VERTEX,
        },
    ));

    wasm_webxr_helpers::Session { inner: xr_session }.run_rendering_loop(move |_time, frame| {
        let xr_session: web_sys::XrSession = frame.session();

        let pose = match frame.get_viewer_pose(&reference_space) {
            Some(pose) => pose,
            None => return,
        };

        let views: Vec<web_sys::XrView> = pose.views().iter().map(|view| view.into()).collect();

        struct Viewport {
            x: f32,
            y: f32,
            width: f32,
            height: f32,
        }

        let viewports: Vec<_> = views
            .iter()
            .map(|view| {
                let viewport = xr_gl_layer.get_viewport(view).unwrap();

                Viewport {
                    x: viewport.x() as f32,
                    y: viewport.y() as f32,
                    width: viewport.width() as f32,
                    height: viewport.height() as f32,
                }
            })
            .collect();

        let base_layer = xr_session.render_state().base_layer().unwrap();

        {
            let parse_matrix = |vec| Mat4::from_cols_array(&<[f32; 16]>::try_from(vec).unwrap());

            let left_proj = parse_matrix(views[0].projection_matrix());
            let left_inv = parse_matrix(views[0].transform().inverse().matrix());

            queue.write_buffer(
                &left_eye_uniform_buffer,
                0,
                bytemuck::bytes_of(
                    &shared_structs::Uniforms {
                        projection_view: { left_proj * left_inv }.into(),
                        eye_position: {
                            let p = views[0].transform().position();
                            glam::DVec3::new(p.x(), p.y(), p.z()).as_vec3()
                        },
                    }
                    .as_std140(),
                ),
            );

            if let Some(right_view) = views.get(1) {
                let right_inv = parse_matrix(right_view.transform().inverse().matrix());
                let right_proj = parse_matrix(right_view.projection_matrix());

                queue.write_buffer(
                    &right_eye_uniform_buffer,
                    0,
                    bytemuck::bytes_of(
                        &shared_structs::Uniforms {
                            projection_view: { right_proj * right_inv }.into(),
                            eye_position: {
                                let p = right_view.transform().position();
                                glam::DVec3::new(p.x(), p.y(), p.z()).as_vec3()
                            },
                        }
                        .as_std140(),
                    ),
                );
            }
        }

        let framebuffer = base_layer.framebuffer();

        let texture = unsafe {
            device.inner.create_texture_from_hal::<wgpu_hal::gles::Api>(
                wgpu_hal::gles::Texture {
                    inner: wgpu_hal::gles::TextureInner::ExternalFramebuffer {
                        inner: framebuffer.clone(),
                    },
                    mip_level_count: 1,
                    array_layer_count: 1,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    format_desc: wgpu_hal::gles::TextureFormatDesc {
                        internal: glow::RGBA,
                        external: glow::RGBA,
                        data_type: glow::UNSIGNED_BYTE,
                    },
                    copy_size: wgpu_hal::CopyExtent {
                        width: base_layer.framebuffer_width(),
                        height: base_layer.framebuffer_height(),
                        depth: 1,
                    },
                },
                &wgpu::TextureDescriptor {
                    label: Some("framebuffer (color)"),
                    size: wgpu::Extent3d {
                        width: base_layer.framebuffer_width(),
                        height: base_layer.framebuffer_height(),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                },
            )
        };

        let depth = unsafe {
            device.inner.create_texture_from_hal::<wgpu_hal::gles::Api>(
                wgpu_hal::gles::Texture {
                    inner: wgpu_hal::gles::TextureInner::ExternalFramebuffer { inner: framebuffer },
                    mip_level_count: 1,
                    array_layer_count: 1,
                    format: wgpu::TextureFormat::Depth32Float,
                    format_desc: wgpu_hal::gles::TextureFormatDesc {
                        internal: glow::RGBA,
                        external: glow::RGBA,
                        data_type: glow::UNSIGNED_BYTE,
                    },
                    copy_size: wgpu_hal::CopyExtent {
                        width: base_layer.framebuffer_width(),
                        height: base_layer.framebuffer_height(),
                        depth: 1,
                    },
                },
                &wgpu::TextureDescriptor {
                    label: Some("framebuffer (depth)"),
                    size: wgpu::Extent3d {
                        width: base_layer.framebuffer_width(),
                        height: base_layer.framebuffer_height(),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                },
            )
        };

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device
            .inner
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("command encoder"),
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main render pass"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if mode == web_sys::XrSessionMode::ImmersiveAr {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        })
                    },
                    store: true,
                },
            }],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });

        let uniform_buffer = |i| {
            BindingResource::Buffer(if i == 0 {
                &left_eye_uniform_buffer
            } else {
                &right_eye_uniform_buffer
            })
        };

        {
            let formats = &[wgpu::TextureFormat::Rgba8Unorm];

            let device = device.with_formats(formats, Some(wgpu::TextureFormat::Depth32Float));

            render_pass.set_pipeline(&pbr_pipeline.pipeline);

            render_pass.set_vertex_buffer(3, instance_buffer.slice(..));

            for primitive in &model.opaque_primitives {
                render_pass.set_vertex_buffer(0, primitive.positions.slice(..));
                render_pass.set_vertex_buffer(1, primitive.normals.slice(..));
                render_pass.set_vertex_buffer(2, primitive.uvs.slice(..));
                render_pass
                    .set_index_buffer(primitive.indices.slice(..), wgpu::IndexFormat::Uint32);

                render_pass.set_bind_group(1, &primitive.bind_group, &[]);

                for (i, viewport) in viewports.iter().enumerate() {
                    render_pass.set_viewport(
                        viewport.x,
                        viewport.y,
                        viewport.width,
                        viewport.height,
                        0.0,
                        1.0,
                    );

                    let bind_group = device.get_bind_group(
                        ("pbr pipeline uniform", i as u32),
                        0,
                        &pbr_pipeline,
                        &[uniform_buffer(i), BindingResource::Sampler(&linear_sampler)],
                    );

                    render_pass.set_bind_group(0, bind_group, &[]);
                    render_pass.draw_indexed(0..primitive.num_indices, 0, 0..1);
                }
            }

            render_pass.set_pipeline(&pbr_alpha_clipped_pipeline.pipeline);

            for primitive in &model.alpha_clipped_primitives {
                render_pass.set_vertex_buffer(0, primitive.positions.slice(..));
                render_pass.set_vertex_buffer(1, primitive.normals.slice(..));
                render_pass.set_vertex_buffer(2, primitive.uvs.slice(..));
                render_pass
                    .set_index_buffer(primitive.indices.slice(..), wgpu::IndexFormat::Uint32);

                render_pass.set_bind_group(1, &primitive.bind_group, &[]);

                for (i, viewport) in viewports.iter().enumerate() {
                    render_pass.set_viewport(
                        viewport.x,
                        viewport.y,
                        viewport.width,
                        viewport.height,
                        0.0,
                        1.0,
                    );
                    let bind_group = device.get_bind_group(
                        ("pbr pipeline alpha clipped uniform", i as u32),
                        0,
                        &pbr_alpha_clipped_pipeline,
                        &[uniform_buffer(i), BindingResource::Sampler(&linear_sampler)],
                    );
                    render_pass.set_bind_group(0, bind_group, &[]);
                    render_pass.draw_indexed(0..primitive.num_indices, 0, 0..1);
                }
            }
        }

        drop(render_pass);

        queue.submit(std::iter::once(encoder.finish()));

        device.flush();
    });
}

fn main() {
    wasm_bindgen_futures::spawn_local(run());
}
