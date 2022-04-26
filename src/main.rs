use crevice::std140::AsStd140;
use futures::FutureExt;
use glam::{Mat4, Vec3};
use kiss_engine_wgpu::{BindingResource, Device, RenderPipelineDesc, ShaderSettings};
use std::borrow::Cow;
use wasm_bindgen::JsCast;
use wasm_webxr_helpers::{button_click_future, create_button};
use wgpu::util::DeviceExt;

mod assets;

use assets::{
    load_gltf_from_url, load_single_pixel_image, prune_fetched_images, FetchedImages,
    ModelLoadContext,
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

    let href = web_sys::window().unwrap().location().href().unwrap();
    let href = url::Url::parse(&href).unwrap();

    let mut model_urls = vec![
        Cow::Borrowed("glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf"),
        Cow::Borrowed("controller_model/controller.gltf"),
        Cow::Borrowed("glTF-Sample-Models/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf"),
    ];

    let mut no_sponza = false;

    for (key, value) in href.query_pairs() {
        log::warn!("{} {}", key, &value);

        if key == "model" {
            model_urls.push(value);
        } else if key == "nosponza" {
            no_sponza = true;
        }
    }

    if no_sponza {
        model_urls.remove(0);
    }

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
        _ => web_sys::XrReferenceSpaceType::LocalFloor,
    };

    let required_features = js_sys::Array::of1(&"local-floor".into());

    let xr_session: web_sys::XrSession =
        wasm_bindgen_futures::JsFuture::from(xr.request_session_with_options(
            mode,
            web_sys::XrSessionInit::new().required_features(&required_features),
        ))
        .await
        .unwrap()
        .into();

    let xr_gl_layer =
        web_sys::XrWebGlLayer::new_with_web_gl2_rendering_context(&xr_session, &webgl2_context)
            .unwrap();

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
        "Using {} with the {:?} backend. Downlevel capabilities: {:?}",
        adapter_info.name,
        adapter_info.backend,
        adapter.get_downlevel_capabilities()
    );

    let supported_features = adapter.features();
    log::info!("Supported features: {:?}", supported_features);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device"),
                features: supported_features,
                limits: adapter.limits(),
            },
            None,
        )
        .await
        .expect("Unable to find a suitable GPU adapter!");

    let mut device = Device::new(device);

    let mut fetched_images = FetchedImages::default();

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

    let pbr_pipeline = kiss_engine_wgpu::create_render_pipeline_with_wgpu_vertex_buffer_layout(
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
            wgpu::VertexBufferLayout {
                array_stride: 3 * 4,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3],
            },
            wgpu::VertexBufferLayout {
                array_stride: 3 * 4,
                attributes: &wgpu::vertex_attr_array![1 => Float32x3],
                step_mode: wgpu::VertexStepMode::Vertex,
            },
            wgpu::VertexBufferLayout {
                array_stride: 2 * 4,
                attributes: &wgpu::vertex_attr_array![2 => Float32x2],
                step_mode: wgpu::VertexStepMode::Vertex,
            },
            wgpu::VertexBufferLayout {
                array_stride: 8 * 4,
                attributes: &wgpu::vertex_attr_array![3 => Float32x4, 4 => Float32x4],
                step_mode: wgpu::VertexStepMode::Instance,
            },
        ],
        &[wgpu::TextureFormat::Rgba8Unorm],
        Some(wgpu::TextureFormat::Depth32Float),
    );

    let pbr_alpha_clipped_pipeline =
        kiss_engine_wgpu::create_render_pipeline_with_wgpu_vertex_buffer_layout(
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
                wgpu::VertexBufferLayout {
                    array_stride: 3 * 4,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                },
                wgpu::VertexBufferLayout {
                    array_stride: 3 * 4,
                    attributes: &wgpu::vertex_attr_array![1 => Float32x3],
                    step_mode: wgpu::VertexStepMode::Vertex,
                },
                wgpu::VertexBufferLayout {
                    array_stride: 2 * 4,
                    attributes: &wgpu::vertex_attr_array![2 => Float32x2],
                    step_mode: wgpu::VertexStepMode::Vertex,
                },
                wgpu::VertexBufferLayout {
                    array_stride: 8 * 4,
                    attributes: &wgpu::vertex_attr_array![3 => Float32x4, 4 => Float32x4],
                    step_mode: wgpu::VertexStepMode::Instance,
                },
            ],
            &[wgpu::TextureFormat::Rgba8Unorm],
            Some(wgpu::TextureFormat::Depth32Float),
        );

    let line_pipeline = kiss_engine_wgpu::create_render_pipeline_with_wgpu_vertex_buffer_layout(
        &device.inner,
        "line pipeline",
        device.get_shader(
            "line_vertex.spv",
            include_bytes!("../line_vertex.spv"),
            ShaderSettings {
                entry_point: "line_vertex",
                ..Default::default()
            },
        ),
        device.get_shader(
            "flat_colour.spv",
            include_bytes!("../flat_colour.spv"),
            ShaderSettings {
                entry_point: "flat_colour",
                ..Default::default()
            },
        ),
        RenderPipelineDesc {
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_compare: wgpu::CompareFunction::Less,
            ..Default::default()
        },
        &[wgpu::VertexBufferLayout {
            array_stride: 6 * 4,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        }],
        &[wgpu::TextureFormat::Rgba8Unorm],
        Some(wgpu::TextureFormat::Depth32Float),
    );

    basis_universal_wasm::wasm_init().await.unwrap();
    basis_universal_wasm::initialize_basis();

    let mut context = ModelLoadContext {
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
        supported_features,
    };

    let mut models = Vec::new();

    let instances = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut instance_counts = Vec::new();

    log::info!("urls: {:?}", model_urls);

    for (i, model_url) in model_urls.iter().enumerate() {
        let url = url::Url::options()
            .base_url(Some(&href))
            .parse(model_url)
            .unwrap();

        models.push(load_gltf_from_url(&url, &mut context).await);
        let mut instances = instances.borrow_mut();
        if i == 1 {
            instance_counts.push(4);
            instances.push(Instance::default());
            instances.push(Instance::default());
            instances.push(Instance::default());
            instances.push(Instance::default());
        } else if i == 2 {
            instance_counts.push(1);
            instances.push(Instance::scaled(0.01));
        } else {
            instance_counts.push(1);
            instances.push(Instance::default());
        }
    }

    let setup_fn: js_sys::Function =
        js_sys::Reflect::get(&web_sys::window().unwrap(), &"setReceiveMessage".into())
            .unwrap()
            .into();

    let send_fn: js_sys::Function =
        js_sys::Reflect::get(&web_sys::window().unwrap(), &"sendMessage".into())
            .unwrap()
            .into();

    let instances_clone = instances.clone();
    let on_message =
        wasm_bindgen::closure::Closure::wrap(Box::new(move |id: u32, ev: web_sys::MessageEvent| {
            let data: js_sys::ArrayBuffer = ev.data().into();
            let uint8 = js_sys::Uint8Array::new(&data);
            let bytes = uint8.to_vec();
            // Bytemuck panics with an alignment error if we try and cast to an instance.
            let floats: &[f32] = bytemuck::cast_slice(&bytes);
            instances_clone.borrow_mut()[5] = Instance::from_slice(floats);
            instances_clone.borrow_mut()[3] = Instance::from_slice(&floats[8..]);
            instances_clone.borrow_mut()[4] = Instance::from_slice(&floats[16..]);
        })
            as Box<dyn FnMut(u32, web_sys::MessageEvent)>);

    setup_fn
        .call1(
            &wasm_bindgen::JsValue::undefined(),
            on_message.as_ref().unchecked_ref(),
        )
        .unwrap();
    // We need do this this as otherwise `on_message` is dropped when `run()` finishes.
    on_message.forget();

    let mut instance_buffer = ResizingBuffer::new(
        &device.inner,
        bytemuck::cast_slice(&instances.borrow()),
        wgpu::BufferUsages::VERTEX,
    );

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

    let mut line_verts = [
        LineVertex {
            position: -Vec3::ONE,
            colour: Vec3::X,
        },
        LineVertex {
            position: Vec3::ONE,
            colour: Vec3::Y,
        },
        LineVertex {
            position: Vec3::new(-1.0, 1.0, -1.0),
            colour: Vec3::Z,
        },
        LineVertex {
            position: -Vec3::new(-1.0, 1.0, -1.0),
            colour: Vec3::ONE - Vec3::Z,
        },
    ];

    let line_buffer = device.create_resource(device.inner.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("line buffer"),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
            contents: bytemuck::cast_slice(&line_verts),
        },
    ));

    wasm_webxr_helpers::Session { inner: xr_session }.run_rendering_loop(move |_time, frame| {
        let xr_session: web_sys::XrSession = frame.session();

        let pose = match frame.get_viewer_pose(&reference_space) {
            Some(pose) => pose,
            None => return,
        };

        let input_sources = xr_session.input_sources();

        for i in 0..input_sources.length() {
            let input_source = input_sources.get(i).unwrap();

            if let Some(grip_space) = input_source.grip_space() {
                let grip_pose = frame.get_pose(&grip_space, &reference_space).unwrap();
                let transform = grip_pose.transform();
                let instance = Instance::from_transform(transform, 1.0);
                instances.borrow_mut()[i as usize + 1] = instance;
                line_verts[i as usize * 2].position = instance.position;
            }
        }

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

            // Send the head transform to remotes.
            {
                let mut head_transform = Instance::from_transform(pose.transform(), 0.35);
                head_transform.rotation *= glam::Quat::from_rotation_y(std::f32::consts::PI);
                let mut array = [0.0; 24];
                head_transform.write_to_slice(&mut array);
                // Write hands.
                instances.borrow()[1].write_to_slice(&mut array[8..]);
                instances.borrow()[2].write_to_slice(&mut array[16..]);
                let bytes = bytemuck::bytes_of(&array);

                let uint8 = unsafe { js_sys::Uint8Array::view(bytes) };

                send_fn
                    .call1(&wasm_bindgen::JsValue::undefined(), &uint8)
                    .unwrap();
            }

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

            instance_buffer.write(
                &device.inner,
                &queue,
                bytemuck::cast_slice(&instances.borrow()),
            );

            queue.write_buffer(&line_buffer, 0, bytemuck::cast_slice(&line_verts));
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
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
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

            render_pass.set_vertex_buffer(3, instance_buffer.inner.slice(..));

            let mut instance_offset = 0;

            for (model_index, model) in models.iter().enumerate() {
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
                        render_pass.draw_indexed(
                            0..primitive.num_indices,
                            0,
                            instance_offset..instance_offset + instance_counts[model_index],
                        );
                    }
                }

                instance_offset += instance_counts[model_index];
            }

            //render_pass.set_pipeline(&pbr_alpha_clipped_pipeline.pipeline);

            let mut instance_offset = 0;

            for (model_index, model) in models.iter().enumerate() {
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
                        render_pass.draw_indexed(
                            0..primitive.num_indices,
                            0,
                            instance_offset..instance_offset + instance_counts[model_index],
                        );
                    }
                }

                instance_offset += instance_counts[model_index];
            }

            {
                render_pass.set_pipeline(&line_pipeline.pipeline);
                render_pass.set_vertex_buffer(0, line_buffer.slice(..));

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
                        ("line pipeline", i as u32),
                        0,
                        &line_pipeline,
                        &[uniform_buffer(i)],
                    );

                    render_pass.set_bind_group(0, bind_group, &[]);
                    render_pass.draw(0..4, 0..1);
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

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Instance {
    pub position: Vec3,
    pub scale: f32,
    pub rotation: glam::Quat,
}

impl Instance {
    pub fn new(position: Vec3, scale: f32, rotation: glam::Quat) -> Self {
        Self {
            position,
            scale,
            rotation,
        }
    }

    pub fn scaled(scale: f32) -> Self {
        let mut instance = Self::default();
        instance.scale = scale;
        instance
    }

    pub fn write_to_slice(&self, slice: &mut [f32]) {
        self.position.write_to_slice(slice);
        slice[3] = self.scale;
        self.rotation.write_to_slice(&mut slice[4..]);
    }

    pub fn from_slice(slice: &[f32]) -> Self {
        Self::new(
            Vec3::from_slice(slice),
            slice[3],
            glam::Quat::from_slice(&slice[4..]),
        )
    }

    pub fn from_transform(transform: web_sys::XrRigidTransform, scale: f32) -> Self {
        let position = transform.position();
        let rotation = transform.orientation();

        let position = glam::DVec3::new(position.x(), position.y(), position.z());
        let rotation =
            glam::DQuat::from_xyzw(rotation.x(), rotation.y(), rotation.z(), rotation.w());
        Self {
            position: position.as_vec3(),
            rotation: rotation.as_f32(),
            scale,
        }
    }
}

impl Default for Instance {
    fn default() -> Self {
        Self::new(Vec3::ZERO, 1.0, glam::Quat::IDENTITY)
    }
}

struct ResizingBuffer {
    capacity: usize,
    inner: wgpu::Buffer,
    usage: wgpu::BufferUsages,
}

impl ResizingBuffer {
    fn new(device: &wgpu::Device, bytes: &[u8], usage: wgpu::BufferUsages) -> Self {
        Self {
            capacity: bytes.len(),
            inner: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage: usage | wgpu::BufferUsages::COPY_DST,
            }),
            usage,
        }
    }

    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) {
        if bytes.len() > self.capacity {
            self.capacity = (self.capacity * 2).max(bytes.len());
            log::warn!("Resizing to {}", self.capacity);
            self.inner = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: self.capacity as u64,
                mapped_at_creation: false,
                usage: self.usage | wgpu::BufferUsages::COPY_DST,
            });
        }

        queue.write_buffer(&self.inner, 0, bytes);
    }
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct LineVertex {
    pub position: Vec3,
    pub colour: Vec3,
}
