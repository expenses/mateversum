use crevice::std140::AsStd140;
use futures::FutureExt;
use glam::{Mat4, Vec3};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsCast;
use wasm_webxr_helpers::{button_click_future, create_button};
use wgpu::util::DeviceExt;

mod assets;
mod caching;

use assets::{
    load_gltf_from_bytes, load_single_pixel_image, FetchedImages, Format, ModelLoadContext,
    ModelPrimitive,
};
use caching::ResourceCache;

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum AnisotrophicFilteringLevel {
    L2 = 2,
    L4 = 4,
    L8 = 8,
    L16 = 16,
}

struct PerformanceSettings {
    anisotrophic_filtering_level: Option<AnisotrophicFilteringLevel>,
    max_texture_size: Option<u32>,
}

impl PerformanceSettings {
    fn anisotropy_clamp(&self) -> Option<std::num::NonZeroU8> {
        self.anisotrophic_filtering_level
            .map(|level| std::num::NonZeroU8::new(level as u8).unwrap())
    }
}

// Code that's ran on init for every thread, both the main thread and web workers.
#[wasm_bindgen(start)]
pub fn main() {
    // todo: fetch the log level from the url like before.
    let _ = console_log::init_with_level(log::Level::Info);
    ::console_error_panic_hook::set_once();

    // Put the basis universal wasm blob into the global scope.
    wasm_bindgen_futures::spawn_local(async move {
        basis_universal_wasm::wasm_init().await;

        // We need to initialize it here because otherwise the basis_universal crate
        // will only let you do so once across all threads.
        basis_universal_wasm::initialize_basis();
    })
}

#[wasm_bindgen]
pub async fn run() -> Result<(), wasm_bindgen::JsValue> {
    let thread_pool = wasm_futures_executor::ThreadPool::max_threads().await?;

    let href = web_sys::window().unwrap().location().href()?;
    let href = url::Url::parse(&href).map_err(|err| err.to_string())?;

    let caches = web_sys::window().unwrap().caches()?;

    let cache: web_sys::Cache = wasm_bindgen_futures::JsFuture::from(caches.open("0.1.0"))
        .await?
        .into();

    let request_client = crate::assets::RequestClient::new(cache).map_err(|err| err.to_string())?;

    let mut model_refs: Vec<ModelReference> = serde_json::from_slice(
        &request_client
            .fetch_bytes_without_caching(
                &url::Url::options()
                    .base_url(Some(&href))
                    .parse("new_sponza.json")
                    .unwrap(),
                None,
            )
            .await
            .map_err(|err| err.to_string())?,
    )
    .map_err(|err| format!("Failed to parse model ref json: {}", err))?;

    for (key, value) in href.query_pairs() {
        if key == "model" {
            model_refs.push(ModelReference {
                url: url::Url::options()
                    .base_url(Some(&href))
                    .parse(&value)
                    .unwrap(),
                position: [0.0; 3],
            });
        }
    }

    let vr_button = create_button("Start VR");
    let ar_button = create_button("Start AR");

    let start_vr_future = button_click_future(&vr_button);
    let start_ar_future = button_click_future(&ar_button);

    let canvas = wasm_webxr_helpers::Canvas::default();
    let webgl2_context = canvas.create_webgl2_context();

    let navigator = web_sys::window().unwrap().navigator();
    let xr = navigator.xr();

    let mode = futures::select! {
        _ = Box::pin(start_vr_future.fuse()) => web_sys::XrSessionMode::ImmersiveVr,
        _ = Box::pin(start_ar_future.fuse()) => web_sys::XrSessionMode::ImmersiveAr,
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
        .await?
        .into();

    let xr_gl_layer =
        web_sys::XrWebGlLayer::new_with_web_gl2_rendering_context(&xr_session, &webgl2_context)?;

    let mut render_state_init = web_sys::XrRenderStateInit::new();
    render_state_init
        .depth_near(0.001)
        .base_layer(Some(&xr_gl_layer));
    xr_session.update_render_state_with_state(&render_state_init);

    let reference_space: web_sys::XrReferenceSpace = wasm_bindgen_futures::JsFuture::from(
        xr_session.request_reference_space(reference_space_type),
    )
    .await?
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

    log::info!("Supported features: {:?}", adapter.features());

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device"),
                features: adapter.features(),
                limits: adapter.limits(),
            },
            None,
        )
        .await
        .expect("Unable to find a suitable GPU adapter!");

    let device = Rc::new(device);
    let queue = Rc::new(queue);

    let fetched_images = FetchedImages::default();

    // Todo: this should be set by the user
    let mut performance_settings = PerformanceSettings {
        anisotrophic_filtering_level: Some(AnisotrophicFilteringLevel::L16),
        max_texture_size: Some(1024),
    };

    // Bring max texture size more down to earth.
    let device_max_texture_size = adapter.limits().max_texture_dimension_2d;
    performance_settings.max_texture_size = Some(
        performance_settings
            .max_texture_size
            .map(|size| size.min(device_max_texture_size))
            .unwrap_or(device_max_texture_size),
    );

    let compressed_texture_format = Format::new_from_features(adapter.features());

    log::info!(
        "Using compressed texture format: {:?}",
        compressed_texture_format
    );

    let linear_sampler = Rc::new(device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        anisotropy_clamp: performance_settings.anisotropy_clamp(),
        ..Default::default()
    }));

    let uniform_entry = |binding, visibility| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        count: None,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    };

    let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        count: None,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
    };

    let sampler_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        count: None,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
    };

    let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("uniform bind group layout"),
        entries: &[
            uniform_entry(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
            sampler_entry(1),
        ],
    });

    let model_bgl = Rc::new(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("model bind group layout"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
                uniform_entry(4, wgpu::ShaderStages::FRAGMENT),
                sampler_entry(5),
                sampler_entry(6),
                sampler_entry(7),
                sampler_entry(8),
            ],
        }),
    );

    let model_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("model pipeline layout"),
        bind_group_layouts: &[&uniform_bgl, &model_bgl],
        push_constant_ranges: &[],
    });

    let vertex_buffers = &[
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
    ];

    let shader_cache = Rc::new(ResourceCache::default());

    let vertex_state = wgpu::VertexState {
        module: shader_cache.get("vertex", || {
            device.create_shader_module(&wgpu::include_spirv!("../vertex.spv"))
        }),
        entry_point: "vertex",
        buffers: vertex_buffers,
    };

    let pbr_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("pbr pipeline"),
        layout: Some(&model_pipeline_layout),
        vertex: vertex_state.clone(),
        fragment: Some(wgpu::FragmentState {
            module: shader_cache.get("fragment", || {
                device.create_shader_module(&wgpu::include_spirv!("../fragment.spv"))
            }),
            entry_point: "fragment",
            targets: &[wgpu::TextureFormat::Rgba8Unorm.into()],
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Front),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            bias: wgpu::DepthBiasState::default(),
            stencil: Default::default(),
        }),
        multisample: Default::default(),
        multiview: Default::default(),
    });

    let pbr_alpha_clipped_pipeline =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pbr alpha clipped pipeline"),
            layout: Some(&model_pipeline_layout),
            vertex: vertex_state,
            fragment: Some(wgpu::FragmentState {
                module: shader_cache.get("fragment_alpha_clipped", || {
                    device.create_shader_module(&wgpu::include_spirv!(
                        "../fragment_alpha_clipped.spv"
                    ))
                }),
                entry_point: "fragment_alpha_clipped",
                targets: &[wgpu::TextureFormat::Rgba8Unorm.into()],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                bias: wgpu::DepthBiasState::default(),
                stencil: Default::default(),
            }),
            multisample: Default::default(),
            multiview: Default::default(),
        });

    let line_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("line pipeline layout"),
        bind_group_layouts: &[&uniform_bgl],
        push_constant_ranges: &[],
    });

    let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("line pipeline"),
        layout: Some(&line_pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader_cache.get("line_vertex", || {
                device.create_shader_module(&wgpu::include_spirv!("../line_vertex.spv"))
            }),
            entry_point: "line_vertex",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 6 * 4,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader_cache.get("flat_colour", || {
                device.create_shader_module(&wgpu::include_spirv!("../flat_colour.spv"))
            }),
            entry_point: "flat_colour",
            targets: &[wgpu::TextureFormat::Rgba8Unorm.into()],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            bias: wgpu::DepthBiasState::default(),
            stencil: Default::default(),
        }),
        multisample: Default::default(),
        multiview: Default::default(),
    });

    let context = Rc::new(ModelLoadContext {
        device: Rc::clone(&device),
        queue: Rc::clone(&queue),
        fetched_images: Rc::new(RefCell::new(fetched_images)),
        model_bgl: Rc::clone(&model_bgl),
        black_image: load_single_pixel_image(
            &*device,
            &queue,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &[0, 0, 0, 255],
        ),
        default_metallic_roughness_image: load_single_pixel_image(
            &*device,
            &queue,
            wgpu::TextureFormat::Rgba8Unorm,
            &[0, 255, 0, 255],
        ),
        flat_normals_image: load_single_pixel_image(
            &*device,
            &queue,
            wgpu::TextureFormat::Rgba8Unorm,
            &[127, 127, 255, 255],
        ),
        white_image: load_single_pixel_image(
            &*device,
            &queue,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &[255, 255, 255, 255],
        ),
        compressed_texture_format,
        shader_cache: Rc::clone(&shader_cache),
        pipeline_cache: Default::default(),
        sampler: Rc::clone(&linear_sampler),
        performance_settings,
        thread_pool,
        request_client,
    });

    let models = Rc::new(RefCell::new(Vec::new()));

    // Deduplicate model references with the same url so they get treated as instances of the same model.

    let mut model_instances: HashMap<url::Url, Vec<Instance>> = HashMap::new();

    for model_ref in model_refs {
        model_instances
            .entry(model_ref.url)
            .or_default()
            .push(Instance::new(
                model_ref.position.into(),
                1.0,
                glam::Quat::IDENTITY,
            ));
    }

    for (model_url, instances) in model_instances {
        let models = Rc::clone(&models);
        let context = Rc::clone(&context);
        let device = Rc::clone(&device);
        wasm_bindgen_futures::spawn_local(async move {
            let bytes = context
                .request_client
                .fetch_bytes(&model_url, None)
                .await
                .unwrap();
            let model = load_gltf_from_bytes(&bytes, Some(model_url.clone()), &context)
                .await
                .unwrap();

            models.borrow_mut().push(InstancedModel {
                model,
                instance_buffer: ResizingBuffer::new(
                    &device,
                    bytemuck::cast_slice(&instances),
                    wgpu::BufferUsages::VERTEX,
                ),
                instances,
            });
        });
    }

    let hand_model = {
        let url = url::Url::options()
            .base_url(Some(&href))
            .parse("controller_model/controller.gltf")
            .map_err(|err| err.to_string())?;

        let bytes = context
            .request_client
            .fetch_bytes(&url, None)
            .await
            .map_err(|err| err.to_string())?;

        load_gltf_from_bytes(&bytes, Some(url), &context)
            .await
            .map_err(|err| err.to_string())?
    };

    let head_model = {
        let url = url::Url::options()
            .base_url(Some(&href))
            .parse("glTF-Sample-Models/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf")
            .map_err(|err| err.to_string())?;

        let bytes = context
            .request_client
            .fetch_bytes(&url, None)
            .await
            .map_err(|err| err.to_string())?;

        load_gltf_from_bytes(&bytes, Some(url), &context)
            .await
            .map_err(|err| err.to_string())?
    };

    let setup_fn: js_sys::Function =
        js_sys::Reflect::get(&web_sys::window().unwrap(), &"set_xr_data_handler".into())?.into();

    let send_fn: js_sys::Function =
        js_sys::Reflect::get(&web_sys::window().unwrap(), &"send_xr_data".into())?.into();

    let player_states = Rc::new(RefCell::new(std::collections::HashMap::new()));
    let on_message = {
        let player_states = Rc::clone(&player_states);

        wasm_bindgen::closure::Closure::wrap(Box::new(
            move |uint8: js_sys::Uint8Array, peer_id: String| {
                let mut bytes = [0; 96];
                uint8.copy_to(&mut bytes);
                if !bytes.is_empty() {
                    // Bytemuck panics with an alignment error if we try and cast to an instance.
                    let instances: &[Instance] = cast_slice(&bytes);
                    player_states.borrow_mut().insert(
                        peer_id,
                        PlayerState {
                            head: instances[0],
                            hands: [instances[1], instances[2]],
                        },
                    );
                } else {
                    log::info!("Got {} bytes; ignoring", bytes.len());
                }
            },
        )
            as Box<dyn FnMut(js_sys::Uint8Array, String)>)
    };

    setup_fn.call1(
        &wasm_bindgen::JsValue::undefined(),
        on_message.as_ref().unchecked_ref(),
    )?;
    // We need do this this as otherwise `on_message` is dropped when `run()` finishes.
    on_message.forget();

    let mut player_heads_buffer =
        ResizingBuffer::new_with_capacity(&device, 4 * 4 * 3, wgpu::BufferUsages::VERTEX);
    let mut player_hands_buffer =
        ResizingBuffer::new_with_capacity(&device, 4 * 4 * 3 * 2, wgpu::BufferUsages::VERTEX);

    let left_eye_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("left eye uniform buffer"),
        size: std::mem::size_of::<shared_structs::Uniforms>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let right_eye_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("right eye uniform buffer"),
        size: std::mem::size_of::<shared_structs::Uniforms>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let left_eye_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("left eye bind group"),
        layout: &uniform_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: left_eye_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&linear_sampler),
            },
        ],
    });

    let right_eye_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("right eye bind group"),
        layout: &uniform_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: right_eye_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&linear_sampler),
            },
        ],
    });

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

    let line_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("line buffer"),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
        contents: bytemuck::cast_slice(&line_verts),
    });

    wasm_webxr_helpers::Session { inner: xr_session }.run_rendering_loop(move |_time, frame| {
        let models = models.borrow();

        let xr_session: web_sys::XrSession = frame.session();

        let pose = match frame.get_viewer_pose(&reference_space) {
            Some(pose) => pose,
            None => return,
        };

        let mut player_state = PlayerState {
            head: Instance::from_transform(pose.transform(), 0.5),
            hands: Default::default(),
        };

        let input_sources = xr_session.input_sources();

        for i in 0..input_sources.length() {
            let input_source = input_sources.get(i).unwrap();

            if let Some(grip_space) = input_source.grip_space() {
                // todo: not sure why this would be None? But has been in 1 case.
                if let Some(grip_pose) = frame.get_pose(&grip_space, &reference_space) {
                    let transform = grip_pose.transform();
                    let instance = Instance::from_transform(transform, 1.0);
                    player_state.hands[i as usize] = instance;
                    line_verts[i as usize * 2].position = instance.position;
                }
            }
        }

        let player_heads: Vec<Instance> = player_states
            .borrow()
            .values()
            .cloned()
            .map(|state| state.head)
            .collect();

        player_heads_buffer.write(&device, &queue, bytemuck::cast_slice(&player_heads));

        let player_hands: Vec<Instance> = std::iter::once(player_state.clone())
            .chain(player_states.borrow().values().cloned())
            .flat_map(|state| state.hands)
            .collect();

        player_hands_buffer.write(&device, &queue, bytemuck::cast_slice(&player_hands));

        let views: Vec<web_sys::XrView> = pose.views().iter().map(|view| view.into()).collect();

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
                let mut head_transform = player_state.head;
                head_transform.rotation *= glam::Quat::from_rotation_y(std::f32::consts::PI);
                let instances = [head_transform, player_state.hands[0], player_state.hands[1]];
                let bytes = bytemuck::cast_slice(&instances);

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

            queue.write_buffer(&line_buffer, 0, bytemuck::cast_slice(&line_verts));
        }

        let framebuffer = base_layer.framebuffer();

        let texture = unsafe {
            device.create_texture_from_hal::<wgpu_hal::gles::Api>(
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
            device.create_texture_from_hal::<wgpu_hal::gles::Api>(
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

        // We borrow all the bind groups here because they need to be borrowed for the entire duration of the render pass.

        let opaque_model_primitives: Vec<Vec<_>> = models
            .iter()
            .map(|model| {
                model
                    .model
                    .opaque_primitives
                    .iter()
                    .map(|primitive| primitive.bind_group.borrow())
                    .collect::<Vec<_>>()
            })
            .collect();

        let alpha_clipped_model_primitives: Vec<Vec<_>> = models
            .iter()
            .map(|model| {
                model
                    .model
                    .alpha_clipped_primitives
                    .iter()
                    .map(|primitive| primitive.bind_group.borrow())
                    .collect::<Vec<_>>()
            })
            .collect();

        let head_primitives = head_model
            .opaque_primitives
            .iter()
            .map(|primitive| primitive.bind_group.borrow())
            .collect::<Vec<_>>();

        let hand_primitives = hand_model
            .opaque_primitives
            .iter()
            .map(|primitive| primitive.bind_group.borrow())
            .collect::<Vec<_>>();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

        let uniform_bind_groups = [&left_eye_bind_group, &right_eye_bind_group];

        {
            render_pass.set_pipeline(&pbr_pipeline);

            for (model_index, model) in models.iter().enumerate() {
                render_pass.set_vertex_buffer(3, model.instance_buffer.inner.slice(..));

                render_primitives(
                    &mut render_pass,
                    &model.model.opaque_primitives,
                    &opaque_model_primitives[model_index],
                    &viewports,
                    &uniform_bind_groups,
                    0..model.instances.len() as u32,
                );
            }

            {
                render_pass.set_vertex_buffer(3, player_heads_buffer.inner.slice(..));
                render_primitives(
                    &mut render_pass,
                    &head_model.opaque_primitives,
                    &head_primitives,
                    &viewports,
                    &uniform_bind_groups,
                    0..player_heads.len() as u32,
                );
            }

            {
                render_pass.set_vertex_buffer(3, player_hands_buffer.inner.slice(..));
                render_primitives(
                    &mut render_pass,
                    &hand_model.opaque_primitives,
                    &hand_primitives,
                    &viewports,
                    &uniform_bind_groups,
                    0..player_hands.len() as u32,
                );
            }

            render_pass.set_pipeline(&pbr_alpha_clipped_pipeline);

            for (model_index, model) in models.iter().enumerate() {
                render_pass.set_vertex_buffer(3, model.instance_buffer.inner.slice(..));

                render_primitives(
                    &mut render_pass,
                    &model.model.alpha_clipped_primitives,
                    &alpha_clipped_model_primitives[model_index],
                    &viewports,
                    &uniform_bind_groups,
                    0..model.instances.len() as u32,
                );
            }

            {
                render_pass.set_pipeline(&line_pipeline);
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

                    render_pass.set_bind_group(0, uniform_bind_groups[i], &[]);
                    render_pass.draw(0..4, 0..1);
                }
            }
        }

        drop(render_pass);

        queue.submit(std::iter::once(encoder.finish()));
    });

    Ok(())
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
        Self {
            scale,
            ..Default::default()
        }
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

    fn new_with_capacity(
        device: &wgpu::Device,
        capacity: usize,
        usage: wgpu::BufferUsages,
    ) -> Self {
        Self {
            capacity,
            inner: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: capacity as u64,
                mapped_at_creation: false,
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

fn cast_slice<F, T>(slice: &[F]) -> &[T] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const T,
            (slice.len() * std::mem::size_of::<F>()) / std::mem::size_of::<T>(),
        )
    }
}

#[derive(Clone)]
struct PlayerState {
    head: Instance,
    hands: [Instance; 2],
}

struct Viewport {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

fn render_primitives<'a>(
    render_pass: &mut wgpu::RenderPass<'a>,
    primitives: &'a [ModelPrimitive],
    bind_groups: &'a [std::cell::Ref<wgpu::BindGroup>],
    viewports: &[Viewport],
    uniform_bind_groups: &[&'a wgpu::BindGroup],
    instance_range: std::ops::Range<u32>,
) {
    for (primitive_index, primitive) in primitives.iter().enumerate() {
        render_pass.set_vertex_buffer(0, primitive.positions.slice(..));
        render_pass.set_vertex_buffer(1, primitive.normals.slice(..));
        render_pass.set_vertex_buffer(2, primitive.uvs.slice(..));
        render_pass.set_index_buffer(primitive.indices.slice(..), wgpu::IndexFormat::Uint32);

        render_pass.set_bind_group(1, &bind_groups[primitive_index], &[]);

        for (i, viewport) in viewports.iter().enumerate() {
            render_pass.set_viewport(
                viewport.x,
                viewport.y,
                viewport.width,
                viewport.height,
                0.0,
                1.0,
            );

            render_pass.set_bind_group(0, uniform_bind_groups[i], &[]);
            render_pass.draw_indexed(0..primitive.num_indices, 0, instance_range.clone());
        }
    }
}

struct InstancedModel {
    model: assets::Model,
    instances: Vec<Instance>,
    instance_buffer: ResizingBuffer,
}

#[derive(serde::Deserialize)]
struct ModelReference {
    url: url::Url,
    position: [f32; 3],
}
