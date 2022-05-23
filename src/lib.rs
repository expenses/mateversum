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

    let mut world: World = serde_json::from_slice(
        &request_client
            .fetch_bytes_without_caching(
                &url::Url::options()
                    .base_url(Some(&href))
                    .parse("sponza_with_mirror.json")
                    .unwrap(),
                None,
            )
            .await
            .map_err(|err| err.to_string())?,
    )
    .map_err(|err| format!("Failed to parse model ref json: {}", err))?;

    for (key, value) in href.query_pairs() {
        if key == "model" {
            world.models.push(ModelReference {
                url: url::Url::options()
                    .base_url(Some(&href))
                    .parse(&value)
                    .unwrap(),
                position: [0.0; 3],
                rotation: [0.0; 3],
                scale: 1.0,
            });
        }
    }

    let vr_button = create_button("Start VR");
    let ar_button = create_button("Start AR");

    append_break();

    let start_vr_future = button_click_future(&vr_button);
    let start_ar_future = button_click_future(&ar_button);

    let canvas = wasm_webxr_helpers::Canvas::default();
    let webgl2_context =
        canvas.create_webgl2_context(wasm_webxr_helpers::ContextCreationOptions { stencil: true });

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

    let multiview = if mode == web_sys::XrSessionMode::ImmersiveVr {
        Some(std::num::NonZeroU32::new(2).unwrap())
    } else {
        None
    };

    let required_features = js_sys::Array::of1(&"local-floor".into());

    let xr_session: web_sys::XrSession =
        wasm_bindgen_futures::JsFuture::from(xr.request_session_with_options(
            mode,
            web_sys::XrSessionInit::new().required_features(&required_features),
        ))
        .await?
        .into();

    let mut layer_init = web_sys::XrWebGlLayerInit::new();

    layer_init.alpha(false).depth(false).stencil(false);

    let xr_gl_layer = web_sys::XrWebGlLayer::new_with_web_gl2_rendering_context_and_layer_init(
        &xr_session,
        &webgl2_context,
        &layer_init,
    )?;

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

    let texture_array_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        count: None,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2Array,
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

    let mirror_uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mirror bind group layout"),
        entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX)],
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

    let tonemap_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mirror bind group layout"),
        entries: &[
            sampler_entry(0),
            if multiview.is_none() {
                texture_entry(1)
            } else {
                texture_array_entry(1)
            },
        ],
    });

    let shader_cache = Rc::new(ResourceCache::default());

    let pipelines = Pipelines::new(
        &device,
        &shader_cache,
        &uniform_bgl,
        &model_bgl,
        &mirror_uniform_bgl,
        &tonemap_bgl,
        multiview,
    );

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

    for model_ref in world.models {
        let instance = model_ref.as_instance();

        model_instances
            .entry(model_ref.url)
            .or_default()
            .push(instance);
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

    let mirror_model = if let Some(model_ref) = &world.mirror {
        let url = model_ref.url.clone();

        let bytes = context
            .request_client
            .fetch_bytes(&url, None)
            .await
            .map_err(|err| err.to_string())?;

        let model = load_gltf_from_bytes(&bytes, Some(url), &context)
            .await
            .map_err(|err| err.to_string())?;

        let instances = vec![model_ref.as_instance()];

        Some(InstancedModel {
            model,
            instance_buffer: ResizingBuffer::new(
                &device,
                bytemuck::cast_slice(&instances),
                wgpu::BufferUsages::VERTEX,
            ),
            instances,
        })
    } else {
        None
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
    let mut player_heads_mirrored_buffer =
        ResizingBuffer::new_with_capacity(&device, 4 * 4 * 3, wgpu::BufferUsages::VERTEX);
    let mut player_hands_buffer =
        ResizingBuffer::new_with_capacity(&device, 4 * 4 * 3 * 2, wgpu::BufferUsages::VERTEX);

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform buffer"),
        size: std::mem::size_of::<shared_structs::Uniforms>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("uniform bind group"),
        layout: &uniform_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
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

    let mirror_instance = world
        .mirror
        .as_ref()
        .map(|model| model.as_instance())
        .unwrap_or_default();

    let mirror_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mirror uniform buffer"),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        contents: bytemuck::bytes_of(
            &shared_structs::MirrorUniforms {
                position: mirror_instance.position,
                normal: mirror_instance.rotation * Vec3::new(0.0, 0.0, -1.0),
            }
            .as_std140(),
        ),
    });

    let mirror_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mirror uniform bind group"),
        layout: &mirror_uniform_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: mirror_uniform_buffer.as_entire_binding(),
        }],
    });

    let movement = Rc::new(RefCell::new(None));

    let on_select_start_closure = wasm_bindgen::closure::Closure::wrap(Box::new({
        let reference_space = reference_space.clone();
        let movement = Rc::clone(&movement);

        move |event: web_sys::XrInputSourceEvent| {
            let frame = event.frame();
            let input_source = event.input_source();

            let target_ray_mode = input_source.target_ray_mode();

            if target_ray_mode == web_sys::XrTargetRayMode::Screen {
                if let Some(grip_pose) =
                    frame.get_pose(&input_source.target_ray_space(), &reference_space)
                {
                    let transform = grip_pose.transform();
                    // Where a tap occurred, and the rotation
                    let instance = Instance::from_transform(transform, 1.0);
                    let tap_direction = instance.rotation * Vec3::Z;

                    log::info!("Starting {:?}", tap_direction);

                    *movement.borrow_mut() = Some(tap_direction);
                }
            }
        }
    })
        as Box<dyn FnMut(web_sys::XrInputSourceEvent)>);

    let on_select_closure = wasm_bindgen::closure::Closure::wrap(Box::new({
        let _reference_space = reference_space.clone();
        let movement = Rc::clone(&movement);

        move |_event: web_sys::XrInputSourceEvent| {
            *movement.borrow_mut() = None;

            /*
            let frame = event.frame();
            let input_source = event.input_source();

            let target_ray_mode = input_source.target_ray_mode();

            if target_ray_mode == web_sys::XrTargetRayMode::Screen {
                if let Some(grip_pose) =
                    frame.get_pose(&input_source.target_ray_space(), &reference_space)
                {
                    let transform = grip_pose.transform();
                    // Where a tap occurred, and the rotation
                    let instance = Instance::from_transform(transform, 1.0);
                    let tap_direction = instance.rotation * Vec3::Z;

                    log::info!("Ending {:?}", tap_direction);
                }
            }
            */
        }
    })
        as Box<dyn FnMut(web_sys::XrInputSourceEvent)>);

    xr_session.set_onselectstart(Some(on_select_start_closure.as_ref().unchecked_ref()));

    xr_session.set_onselect(Some(on_select_closure.as_ref().unchecked_ref()));

    on_select_closure.forget();
    on_select_start_closure.forget();

    let framebuffer_cache = ResourceCache::default();
    let bind_group_cache = ResourceCache::default();

    let mut offset = Vec3::ZERO;

    wasm_webxr_helpers::Session { inner: xr_session }.run_rendering_loop(move |_time, frame| {
        let movement = movement.borrow();

        if let Some(movement) = movement.as_ref() {
            offset += *movement * 2.0 / 60.0;
        }

        let mut js_offset = web_sys::DomPointInit::new();

        js_offset
            .x(offset.x as f64)
            .y(offset.y as f64)
            .z(offset.z as f64);

        let reference_space = reference_space.get_offset_reference_space(
            &web_sys::XrRigidTransform::new_with_position(&js_offset).unwrap(),
        );

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

        let player_heads_mirrored: Vec<Instance> = std::iter::once(player_state.clone())
            .map(|state| {
                let mut head_transform = state.head;

                head_transform.rotation *= glam::Quat::from_rotation_y(std::f32::consts::PI);
                head_transform
            })
            .chain(
                player_states
                    .borrow()
                    .values()
                    .cloned()
                    .map(|state| state.head),
            )
            .collect();

        player_heads_mirrored_buffer.write(
            &device,
            &queue,
            bytemuck::cast_slice(&player_heads_mirrored),
        );

        let player_hands: Vec<Instance> = std::iter::once(player_state.clone())
            .chain(player_states.borrow().values().cloned())
            .flat_map(|state| state.hands)
            .collect();

        player_hands_buffer.write(&device, &queue, bytemuck::cast_slice(&player_hands));

        let views: Vec<web_sys::XrView> = pose.views().iter().map(|view| view.into()).collect();

        let base_layer = xr_session.render_state().base_layer().unwrap();

        {
            let parse_matrix = |vec| Mat4::from_cols_array(&<[f32; 16]>::try_from(vec).unwrap());

            let left_proj = parse_matrix(views[0].projection_matrix());
            let left_inv = parse_matrix(views[0].transform().inverse().matrix());

            let left_projection_view = (left_proj * left_inv).into();
            let left_eye_position = {
                let p = views[0].transform().position();
                glam::DVec3::new(p.x(), p.y(), p.z()).as_vec3()
            };

            let (right_projection_view, right_eye_position) = if let Some(right_view) = views.get(1)
            {
                let right_inv = parse_matrix(right_view.transform().inverse().matrix());
                let right_proj = parse_matrix(right_view.projection_matrix());

                ((right_proj * right_inv).into(), {
                    let p = right_view.transform().position();
                    glam::DVec3::new(p.x(), p.y(), p.z()).as_vec3()
                })
            } else {
                Default::default()
            };

            queue.write_buffer(
                &uniform_buffer,
                0,
                bytemuck::bytes_of(
                    &shared_structs::Uniforms {
                        left_projection_view,
                        right_projection_view,
                        left_eye_position,
                        right_eye_position,
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

            queue.write_buffer(&line_buffer, 0, bytemuck::cast_slice(&line_verts));
        }

        let framebuffer = js_sys::Reflect::get(&base_layer, &"framebuffer".into())
            .unwrap()
            .into();

        let texture = unsafe {
            device.create_texture_from_hal::<wgpu_hal::gles::Api>(
                wgpu_hal::gles::Texture {
                    inner: wgpu_hal::gles::TextureInner::ExternalFramebuffer { inner: framebuffer },
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

        let num_views = multiview.map(|views| views.get()).unwrap_or(1);

        let hdr_framebuffer = framebuffer_cache.get("hdr framebuffer", || {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("hdr framebuffer"),
                    size: wgpu::Extent3d {
                        width: base_layer.framebuffer_width() / num_views,
                        height: base_layer.framebuffer_height(),
                        depth_or_array_layers: num_views,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                })
                .create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(if multiview.is_none() {
                        wgpu::TextureViewDimension::D2
                    } else {
                        wgpu::TextureViewDimension::D2Array
                    }),
                    ..Default::default()
                })
        });

        let tonemap_bind_group = bind_group_cache.get("tonemap bind group", || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tonemap bind group"),
                layout: &tonemap_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(&linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&hdr_framebuffer),
                    },
                ],
            })
        });

        let depth = framebuffer_cache.get("depth", || {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("depth"),
                    size: wgpu::Extent3d {
                        width: base_layer.framebuffer_width() / num_views,
                        height: base_layer.framebuffer_height(),
                        depth_or_array_layers: num_views,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                })
                .create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(if multiview.is_none() {
                        wgpu::TextureViewDimension::D2
                    } else {
                        wgpu::TextureViewDimension::D2Array
                    }),
                    ..Default::default()
                })
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // We borrow all the bind groups here because they need to be borrowed for the entire duration of the render pass.

        let model_bind_groups = ModelBindGroups {
            pbr_opaque: models
                .iter()
                .map(|model| {
                    model
                        .model
                        .opaque_primitives
                        .iter()
                        .map(|primitive| primitive.bind_group.borrow())
                        .collect::<Vec<_>>()
                })
                .collect(),
            pbr_alpha_clipped: models
                .iter()
                .map(|model| {
                    model
                        .model
                        .alpha_clipped_primitives
                        .iter()
                        .map(|primitive| primitive.bind_group.borrow())
                        .collect::<Vec<_>>()
                })
                .collect(),
            unlit_opaque: models
                .iter()
                .map(|model| {
                    model
                        .model
                        .unlit_opaque_primitives
                        .iter()
                        .map(|primitive| primitive.bind_group.borrow())
                        .collect::<Vec<_>>()
                })
                .collect(),
            unlit_alpha_clipped: models
                .iter()
                .map(|model| {
                    model
                        .model
                        .unlit_alpha_clipped_primitives
                        .iter()
                        .map(|primitive| primitive.bind_group.borrow())
                        .collect::<Vec<_>>()
                })
                .collect(),
        };

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

        let mirror_primitives = mirror_model
            .iter()
            .flat_map(|model| &model.model.opaque_primitives)
            .map(|primitive| primitive.bind_group.borrow())
            .collect::<Vec<_>>();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("command encoder"),
        });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main render pass"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &hdr_framebuffer,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1 / 5.0,
                        g: 0.2 / 5.0,
                        b: 0.3 / 5.0,
                        a: 1.0,
                    }),
                    store: true,
                },
            }],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0),
                    store: true,
                }),
            }),
        });

        let heads = HeadOrHandsRenderingData {
            instances: &player_heads_buffer.inner,
            model: &head_model,
            bind_groups: &head_primitives,
            num_instances: player_heads.len() as u32,
        };

        let heads_mirrored = HeadOrHandsRenderingData {
            instances: &player_heads_mirrored_buffer.inner,
            model: &head_model,
            bind_groups: &head_primitives,
            num_instances: player_heads_mirrored.len() as u32,
        };

        let hands = HeadOrHandsRenderingData {
            instances: &player_hands_buffer.inner,
            model: &hand_model,
            bind_groups: &hand_primitives,
            num_instances: player_hands.len() as u32,
        };

        render_pass.set_bind_group(0, &uniform_bind_group, &[]);

        if let Some(mirror_model) = &mirror_model {
            render_pass.set_stencil_reference(1);

            {
                render_pass.set_pipeline(&pipelines.stencil_write);

                render_pass.set_vertex_buffer(3, mirror_model.instance_buffer.inner.slice(..));
                render_primitives(
                    &mut render_pass,
                    &mirror_model.model.opaque_primitives,
                    &mirror_primitives,
                    0..1,
                );
            }

            render_pass.set_bind_group(2, &mirror_uniform_bind_group, &[]);

            {
                render_pass.set_pipeline(&pipelines.pbr.opaque_mirrored);

                render_all(
                    &mut render_pass,
                    &models,
                    |model| &model.opaque_primitives,
                    &model_bind_groups.pbr_opaque,
                    &heads_mirrored,
                    &hands,
                );

                render_pass.set_pipeline(&pipelines.unlit.opaque_mirrored);

                render_all(
                    &mut render_pass,
                    &models,
                    |model| &model.unlit_opaque_primitives,
                    &model_bind_groups.unlit_opaque,
                    &heads_mirrored,
                    &hands,
                );

                render_pass.set_pipeline(&pipelines.pbr.alpha_clipped_mirrored);

                render_all(
                    &mut render_pass,
                    &models,
                    |model| &model.alpha_clipped_primitives,
                    &model_bind_groups.pbr_alpha_clipped,
                    &heads_mirrored,
                    &hands,
                );

                render_pass.set_pipeline(&pipelines.unlit.alpha_clipped_mirrored);

                render_all(
                    &mut render_pass,
                    &models,
                    |model| &model.unlit_alpha_clipped_primitives,
                    &model_bind_groups.unlit_alpha_clipped,
                    &heads_mirrored,
                    &hands,
                );
            }

            {
                render_pass.set_pipeline(&pipelines.set_depth);

                render_pass.set_vertex_buffer(3, mirror_model.instance_buffer.inner.slice(..));
                render_primitives(
                    &mut render_pass,
                    &mirror_model.model.opaque_primitives,
                    &mirror_primitives,
                    0..1,
                );
            }
        }

        {
            render_pass.set_pipeline(&pipelines.pbr.opaque);

            render_all(
                &mut render_pass,
                &models,
                |model| &model.opaque_primitives,
                &model_bind_groups.pbr_opaque,
                &heads,
                &hands,
            );

            render_pass.set_pipeline(&pipelines.unlit.opaque);

            render_all(
                &mut render_pass,
                &models,
                |model| &model.unlit_opaque_primitives,
                &model_bind_groups.unlit_opaque,
                &heads,
                &hands,
            );

            render_pass.set_pipeline(&pipelines.pbr.alpha_clipped);

            render_all(
                &mut render_pass,
                &models,
                |model| &model.alpha_clipped_primitives,
                &model_bind_groups.pbr_alpha_clipped,
                &heads,
                &hands,
            );

            render_pass.set_pipeline(&pipelines.unlit.alpha_clipped);

            render_all(
                &mut render_pass,
                &models,
                |model| &model.unlit_alpha_clipped_primitives,
                &model_bind_groups.unlit_alpha_clipped,
                &heads,
                &hands,
            );
        }

        {
            render_pass.set_pipeline(&pipelines.line);
            render_pass.set_vertex_buffer(0, line_buffer.slice(..));
            render_pass.draw(0..4, 0..1);
        }

        drop(render_pass);

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tonemap render pass"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        render_pass.set_pipeline(&pipelines.tonemap);

        render_pass.set_bind_group(0, tonemap_bind_group, &[]);

        render_pass.draw(0..3, 0..1);

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

fn render_primitives<'a>(
    render_pass: &mut wgpu::RenderPass<'a>,
    primitives: &'a [ModelPrimitive],
    bind_groups: &'a [std::cell::Ref<wgpu::BindGroup>],
    instance_range: std::ops::Range<u32>,
) {
    for (primitive_index, primitive) in primitives.iter().enumerate() {
        render_pass.set_vertex_buffer(0, primitive.positions.slice(..));
        render_pass.set_vertex_buffer(1, primitive.normals.slice(..));
        render_pass.set_vertex_buffer(2, primitive.uvs.slice(..));
        render_pass.set_index_buffer(primitive.indices.slice(..), wgpu::IndexFormat::Uint32);

        render_pass.set_bind_group(1, &bind_groups[primitive_index], &[]);
        render_pass.draw_indexed(0..primitive.num_indices, 0, instance_range.clone());
    }
}

struct ModelBindGroups<'a> {
    pbr_opaque: Vec<Vec<std::cell::Ref<'a, wgpu::BindGroup>>>,
    pbr_alpha_clipped: Vec<Vec<std::cell::Ref<'a, wgpu::BindGroup>>>,
    unlit_opaque: Vec<Vec<std::cell::Ref<'a, wgpu::BindGroup>>>,
    unlit_alpha_clipped: Vec<Vec<std::cell::Ref<'a, wgpu::BindGroup>>>,
}

struct HeadOrHandsRenderingData<'a> {
    instances: &'a wgpu::Buffer,
    model: &'a assets::Model,
    bind_groups: &'a [std::cell::Ref<'a, wgpu::BindGroup>],
    num_instances: u32,
}

fn render_all<'a, F: Fn(&'a assets::Model) -> &'a [ModelPrimitive]>(
    render_pass: &mut wgpu::RenderPass<'a>,
    models: &'a [InstancedModel],
    primitives_getter: F,
    bind_groups: &'a [Vec<std::cell::Ref<wgpu::BindGroup>>],
    heads: &HeadOrHandsRenderingData<'a>,
    hands: &HeadOrHandsRenderingData<'a>,
) {
    for (model_index, model) in models.iter().enumerate() {
        render_pass.set_vertex_buffer(3, model.instance_buffer.inner.slice(..));

        render_primitives(
            render_pass,
            primitives_getter(&model.model),
            &bind_groups[model_index],
            0..model.instances.len() as u32,
        );
    }

    render_pass.set_vertex_buffer(3, heads.instances.slice(..));

    render_primitives(
        render_pass,
        primitives_getter(heads.model),
        heads.bind_groups,
        0..heads.num_instances,
    );

    render_pass.set_vertex_buffer(3, hands.instances.slice(..));

    render_primitives(
        render_pass,
        primitives_getter(hands.model),
        hands.bind_groups,
        0..hands.num_instances,
    );
}

struct InstancedModel {
    model: assets::Model,
    instances: Vec<Instance>,
    instance_buffer: ResizingBuffer,
}

#[derive(serde::Deserialize)]
struct World {
    models: Vec<ModelReference>,
    mirror: Option<ModelReference>,
}

#[derive(serde::Deserialize)]
struct ModelReference {
    #[serde(deserialize_with = "deserialize_relative_url")]
    url: url::Url,
    #[serde(default)]
    position: [f32; 3],
    #[serde(default)]
    rotation: [f32; 3],
    #[serde(default = "one")]
    scale: f32,
}

impl ModelReference {
    fn as_instance(&self) -> Instance {
        Instance::new(
            self.position.into(),
            self.scale,
            glam::Quat::from_euler(
                glam::EulerRot::XYZ,
                self.rotation[0].to_radians(),
                self.rotation[1].to_radians(),
                self.rotation[2].to_radians(),
            ),
        )
    }
}

const fn one() -> f32 {
    1.0
}

fn deserialize_relative_url<'de, D>(deserializer: D) -> Result<url::Url, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;

    let relative = String::deserialize(deserializer)?;

    let href = web_sys::window()
        .unwrap()
        .location()
        .href()
        .map_err(|js_err| serde::de::Error::custom(format!("{:?}", js_err)))?;
    let href = url::Url::parse(&href).map_err(serde::de::Error::custom)?;

    url::Url::options()
        .base_url(Some(&href))
        .parse(&relative)
        .map_err(serde::de::Error::custom)
}

struct PipelineSet {
    opaque: wgpu::RenderPipeline,
    alpha_clipped: wgpu::RenderPipeline,
    opaque_mirrored: wgpu::RenderPipeline,
    alpha_clipped_mirrored: wgpu::RenderPipeline,
}

impl PipelineSet {
    fn new(
        device: &wgpu::Device,
        pipeline_layout: &wgpu::PipelineLayout,
        mirrored_pipeline_layout: &wgpu::PipelineLayout,
        normal_vertex: wgpu::VertexState,
        mirrored_vertex: wgpu::VertexState,
        opaque_fragment: wgpu::FragmentState,
        alpha_clipped_fragment: wgpu::FragmentState,
        multiview: Option<std::num::NonZeroU32>,
    ) -> Self {
        let normal_primitive_state = wgpu::PrimitiveState {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        };

        let normal_depth_state = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            bias: wgpu::DepthBiasState::default(),
            stencil: wgpu::StencilState::default(),
        };

        let stencil_test = wgpu::StencilFaceState {
            compare: wgpu::CompareFunction::Equal,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::Keep,
        };

        let stencil_test_depth_state = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            bias: wgpu::DepthBiasState::default(),
            stencil: wgpu::StencilState {
                front: stencil_test,
                back: stencil_test,
                read_mask: 0xff,
                write_mask: 0xff,
            },
        };

        let mirrored_primitive_state = wgpu::PrimitiveState {
            front_face: wgpu::FrontFace::Cw,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        };

        Self {
            opaque: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(pipeline_layout),
                vertex: normal_vertex.clone(),
                fragment: Some(opaque_fragment.clone()),
                primitive: normal_primitive_state,
                depth_stencil: Some(normal_depth_state.clone()),
                multisample: Default::default(),
                multiview,
            }),
            alpha_clipped: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(pipeline_layout),
                vertex: normal_vertex.clone(),
                fragment: Some(alpha_clipped_fragment.clone()),
                primitive: normal_primitive_state,
                depth_stencil: Some(normal_depth_state),
                multisample: Default::default(),
                multiview,
            }),
            opaque_mirrored: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(mirrored_pipeline_layout),
                vertex: mirrored_vertex.clone(),
                fragment: Some(opaque_fragment.clone()),
                primitive: mirrored_primitive_state,
                depth_stencil: Some(stencil_test_depth_state.clone()),
                multisample: Default::default(),
                multiview,
            }),
            alpha_clipped_mirrored: device.create_render_pipeline(
                &wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(mirrored_pipeline_layout),
                    vertex: mirrored_vertex,
                    fragment: Some(alpha_clipped_fragment),
                    primitive: mirrored_primitive_state,
                    depth_stencil: Some(stencil_test_depth_state),
                    multisample: Default::default(),
                    multiview,
                },
            ),
        }
    }
}

struct Pipelines {
    pbr: PipelineSet,
    unlit: PipelineSet,
    line: wgpu::RenderPipeline,
    stencil_write: wgpu::RenderPipeline,
    set_depth: wgpu::RenderPipeline,
    tonemap: wgpu::RenderPipeline,
}

impl Pipelines {
    fn new(
        device: &wgpu::Device,
        shader_cache: &ResourceCache<wgpu::ShaderModule>,
        uniform_bgl: &wgpu::BindGroupLayout,
        model_bgl: &wgpu::BindGroupLayout,
        mirror_uniform_bgl: &wgpu::BindGroupLayout,
        tonemap_bgl: &wgpu::BindGroupLayout,
        multiview: Option<std::num::NonZeroU32>,
    ) -> Self {
        let model_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("model pipeline layout"),
                bind_group_layouts: &[uniform_bgl, model_bgl],
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

        let prefix = if multiview.is_none() { "single_view::" } else { "" };

        let vertex_state = wgpu::VertexState {
            module: shader_cache.get("vertex", || {
                device.create_shader_module(&if multiview.is_none() {
                    wgpu::include_spirv!("../compiled-shaders/single_view_vertex.spv")
                } else {
                    wgpu::include_spirv!("../compiled-shaders/vertex.spv")
                })
            }),
            entry_point: &format!("{}vertex", prefix),
            buffers: vertex_buffers,
        };

        let normal_primitive_state = wgpu::PrimitiveState {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        };

        let normal_depth_state = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            bias: wgpu::DepthBiasState::default(),
            stencil: wgpu::StencilState::default(),
        };

        let line_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("line pipeline layout"),
            bind_group_layouts: &[uniform_bgl],
            push_constant_ranges: &[],
        });

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line pipeline"),
            layout: Some(&line_pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader_cache.get("line_vertex", || {
                    device.create_shader_module(&if multiview.is_none() {
                        wgpu::include_spirv!("../compiled-shaders/single_view_line_vertex.spv")
                    } else {
                        wgpu::include_spirv!("../compiled-shaders/line_vertex.spv")
                    })
                }),
                entry_point: &format!("{}line_vertex", prefix),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 6 * 4,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader_cache.get("flat_colour", || {
                    device.create_shader_module(&wgpu::include_spirv!(
                        "../compiled-shaders/flat_colour.spv"
                    ))
                }),
                entry_point: "flat_colour",
                targets: &[wgpu::TextureFormat::Rgba16Float.into()],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(normal_depth_state.clone()),
            multisample: Default::default(),
            multiview,
        });

        let tonemap_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[tonemap_bgl],
                push_constant_ranges: &[],
            });

        let tonemap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tonemap pipeline"),
            layout: Some(&tonemap_pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader_cache.get("fullscreen_tri", || {
                    device.create_shader_module(&wgpu::include_spirv!(
                        "../compiled-shaders/fullscreen_tri.spv"
                    ))
                }),
                entry_point: "fullscreen_tri",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader_cache.get("tonemap", || {
                    device.create_shader_module(&if multiview.is_none() {
                        wgpu::include_spirv!("../compiled-shaders/single_view_tonemap.spv")
                    } else {
                        wgpu::include_spirv!("../compiled-shaders/tonemap.spv")
                    })
                }),
                entry_point: &format!("{}tonemap", prefix),
                targets: &[wgpu::TextureFormat::Rgba8Unorm.into()],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: Default::default(),
        });

        let stencil_write = wgpu::StencilFaceState {
            compare: wgpu::CompareFunction::Always,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::IncrementClamp,
        };

        let stencil_write_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("stencil write pipeline"),
                layout: Some(&model_pipeline_layout),
                vertex: vertex_state.clone(),

                fragment: Some(wgpu::FragmentState {
                    module: shader_cache.get("flat_blue", || {
                        device.create_shader_module(&wgpu::include_spirv!(
                            "../compiled-shaders/flat_blue.spv"
                        ))
                    }),
                    entry_point: "flat_blue",
                    targets: &[wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::empty(),
                    }],
                }),
                primitive: normal_primitive_state,
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    bias: wgpu::DepthBiasState::default(),
                    stencil: wgpu::StencilState {
                        front: stencil_write,
                        back: stencil_write,
                        read_mask: 0xff,
                        write_mask: 0xff,
                    },
                }),
                multisample: Default::default(),
                multiview,
            });

        let set_depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("set depth pipeline"),
            layout: Some(&model_pipeline_layout),
            vertex: vertex_state.clone(),

            fragment: Some(wgpu::FragmentState {
                module: shader_cache.get("flat_blue", || {
                    device.create_shader_module(&wgpu::include_spirv!(
                        "../compiled-shaders/flat_blue.spv"
                    ))
                }),
                entry_point: "flat_blue",
                targets: &[wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                }],
            }),
            primitive: normal_primitive_state,
            depth_stencil: Some(normal_depth_state),
            multisample: Default::default(),
            multiview,
        });

        let mirrored_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mirrored pipeline layout"),
                bind_group_layouts: &[uniform_bgl, model_bgl, mirror_uniform_bgl],
                push_constant_ranges: &[],
            });

        let mirrored_vertex = wgpu::VertexState {
            module: shader_cache.get("vertex_mirrored", || {
                device.create_shader_module(&if multiview.is_none() {
                    wgpu::include_spirv!("../compiled-shaders/single_view_vertex_mirrored.spv")
                } else {
                    wgpu::include_spirv!("../compiled-shaders/vertex_mirrored.spv")
                })
            }),
            entry_point: &format!("{}vertex_mirrored", prefix),
            buffers: vertex_buffers,
        };

        Self {
            pbr: PipelineSet::new(
                device,
                &model_pipeline_layout,
                &mirrored_pipeline_layout,
                vertex_state.clone(),
                mirrored_vertex.clone(),
                wgpu::FragmentState {
                    module: shader_cache.get("fragment", || {
                        device.create_shader_module(&if multiview.is_none() {
                            wgpu::include_spirv!("../compiled-shaders/single_view_fragment.spv")
                        } else {
                            wgpu::include_spirv!("../compiled-shaders/fragment.spv")
                        })
                    }),
                    entry_point: &format!("{}fragment", prefix),
                    targets: &[wgpu::TextureFormat::Rgba16Float.into()],
                },
                wgpu::FragmentState {
                    module: shader_cache.get("fragment_alpha_clipped", || {
                        device.create_shader_module(&if multiview.is_none() {
                            wgpu::include_spirv!(
                                "../compiled-shaders/single_view_fragment_alpha_clipped.spv"
                            )
                        } else {
                            wgpu::include_spirv!("../compiled-shaders/fragment_alpha_clipped.spv")
                        })
                    }),
                    entry_point: &format!("{}fragment_alpha_clipped", prefix),
                    targets: &[wgpu::TextureFormat::Rgba16Float.into()],
                },
                multiview,
            ),
            line: line_pipeline,
            stencil_write: stencil_write_pipeline,
            set_depth: set_depth_pipeline,
            tonemap: tonemap_pipeline,
            unlit: PipelineSet::new(
                device,
                &model_pipeline_layout,
                &mirrored_pipeline_layout,
                vertex_state,
                mirrored_vertex,
                wgpu::FragmentState {
                    module: shader_cache.get("fragment_unlit", || {
                        device.create_shader_module(&wgpu::include_spirv!(
                            "../compiled-shaders/fragment_unlit.spv"
                        ))
                    }),
                    entry_point: "fragment_unlit",
                    targets: &[wgpu::TextureFormat::Rgba16Float.into()],
                },
                wgpu::FragmentState {
                    module: shader_cache.get("fragment_unlit_alpha_clipped", || {
                        device.create_shader_module(&wgpu::include_spirv!(
                            "../compiled-shaders/fragment_unlit_alpha_clipped.spv"
                        ))
                    }),
                    entry_point: "fragment_unlit_alpha_clipped",
                    targets: &[wgpu::TextureFormat::Rgba16Float.into()],
                },
                multiview,
            ),
        }
    }
}

pub fn append_break() {
    let br: web_sys::HtmlBrElement = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("br")
        .unwrap()
        .unchecked_into();

    let body = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .body()
        .unwrap();

    body.append_child(&web_sys::Element::from(br)).unwrap();
}
