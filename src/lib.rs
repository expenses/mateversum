use crevice::std140::AsStd140;
use futures::FutureExt;
use glam::{Mat4, Vec3};
use renderer_core::{create_view_from_device_framebuffer, run_rendering_loop};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Range;
use std::rc::Rc;
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsCast;
use wgpu::util::DeviceExt;

mod assets;
mod buffers;
mod caching;
mod js_helpers;
mod pipelines;

use assets::{
    load_gltf_from_bytes, load_single_pixel_image, FetchedImages, Format, ModelLoadContext,
    ModelPrimitive,
};
use buffers::{IndexBuffer, InstanceBuffer, VertexBuffers};
use caching::ResourceCache;
use js_helpers::{append_break, button_click_future, create_button};
use pipelines::{PipelineOptions, Pipelines};

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
    #[cfg(feature = "thread_pool")]
    let thread_pool = wasm_futures_executor::ThreadPool::max_threads().await?;

    let href = web_sys::window().unwrap().location().href()?;
    let href = url::Url::parse(&href).map_err(|err| err.to_string())?;

    let caches = web_sys::window().unwrap().caches()?;

    let cache: web_sys::Cache = wasm_bindgen_futures::JsFuture::from(caches.open("0.1.0"))
        .await?
        .into();

    let request_client = crate::assets::RequestClient::new(cache).map_err(|err| err.to_string())?;

    let mut world_url = url::Url::options()
        .base_url(Some(&href))
        .parse("sponza_with_mirror.json")
        .unwrap();

    for (key, value) in href.query_pairs() {
        if key == "world" {
            world_url = url::Url::options()
                .base_url(Some(&href))
                .parse(&value)
                .unwrap();
        }
    }

    let world: World = serde_json::from_slice(
        &request_client
            .fetch_bytes_without_caching(&world_url, None)
            .await
            .map_err(|err| err.to_string())?,
    )
    .map_err(|err| format!("Failed to parse model ref json: {}", err))?;

    let vr_button = create_button("Start VR");
    let ar_button = create_button("Start AR");

    append_break();

    let start_vr_future = button_click_future(&vr_button);
    let start_ar_future = button_click_future(&ar_button);

    let canvas = renderer_core::Canvas::default();

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

    // Some performance settings.

    let multiview;
    // Whether we're using a fp16 HDR framebuffer that we then tonemap or doing the tonemapping at the end of fragment shaders.
    // Turning this off makes rendering faster at the cost of not being able to do nice post processing effects like bloom.
    let inline_tonemapping;
    let render_skybox;

    if mode == web_sys::XrSessionMode::ImmersiveVr {
        multiview = Some(std::num::NonZeroU32::new(2).unwrap());

        inline_tonemapping = false;
        render_skybox = true;
    } else {
        multiview = None;
        inline_tonemapping = true;
        render_skybox = false;
    };

    let render_direct_to_framebuffer = multiview.is_none() && inline_tonemapping;

    let required_features = js_sys::Array::of1(&"local-floor".into());

    let xr_session: web_sys::XrSession =
        wasm_bindgen_futures::JsFuture::from(xr.request_session_with_options(
            mode,
            web_sys::XrSessionInit::new().required_features(&required_features),
        ))
        .await?
        .into();

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
        "Using '{}' with the {:?} backend. Downlevel capabilities: {:?}",
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

    let mut layer_init = web_sys::XrWebGlLayerInit::new();

    layer_init
        // Setting this crashes the Oculus Quest 2. No idea why. I spent 2+ hours on a git bisect to figure this out.
        // .alpha(false)
        .depth(render_direct_to_framebuffer)
        .stencil(render_direct_to_framebuffer);

    let webgl2_context =
        canvas.create_webgl2_context(renderer_core::ContextCreationOptions { stencil: true });

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

    let cubemap_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        count: None,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::Cube,
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
            texture_entry(2),
            cubemap_entry(3),
            cubemap_entry(4),
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

    let ui_texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ui texture bind group layout"),
        entries: &[texture_entry(0)],
    });

    let skybox_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("skybox bind group layout"),
        entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX)],
    });

    let shader_cache = Rc::new(ResourceCache::default());

    let pipelines = Pipelines::new(
        &device,
        &shader_cache,
        &uniform_bgl,
        &model_bgl,
        &mirror_uniform_bgl,
        &tonemap_bgl,
        &ui_texture_bgl,
        &skybox_bgl,
        &PipelineOptions {
            multiview,
            flip_viewport: render_direct_to_framebuffer,
            inline_tonemapping,
        },
    );

    let vertex_buffers = Rc::new(RefCell::new(VertexBuffers::new(1024, &device)));
    let index_buffer = Rc::new(RefCell::new(IndexBuffer::new(1024, &device)));

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
        #[cfg(feature = "thread_pool")]
        thread_pool,
        request_client,
        bc6h_supported: adapter
            .features()
            .contains(wgpu::Features::TEXTURE_COMPRESSION_BC),
        vertex_buffers: Rc::clone(&vertex_buffers),
        index_buffer: Rc::clone(&index_buffer),
    });

    let models = Rc::new(RefCell::new(slotmap::SlotMap::new()));
    let urls_to_model_slots = Rc::new(RefCell::new(HashMap::new()));

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

        let slot = models.borrow_mut().insert(InstancedModel {
            model: None,
            instances,
        });

        urls_to_model_slots
            .borrow_mut()
            .insert(model_url.as_str().to_string(), slot);

        wasm_bindgen_futures::spawn_local(async move {
            if let Err(error) = load_model(model_url, slot, context, &models).await {
                log::error!("Failed to load model: {}", error);
            }
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

        Some(InstancedModel {
            model: Some(model),
            instances: vec![model_ref.as_instance()],
        })
    } else {
        None
    };

    let ui_plane_model = {
        let url = url::Url::options()
            .base_url(Some(&href))
            .parse("ui/ui_plane.gltf")
            .unwrap();

        let bytes = context
            .request_client
            .fetch_bytes(&url, None)
            .await
            .map_err(|err| err.to_string())?;

        let model = load_gltf_from_bytes(&bytes, Some(url), &context)
            .await
            .map_err(|err| err.to_string())?;

        InstancedModel {
            model: Some(model),
            instances: vec![Instance::default()],
        }
    };

    let ibl_lut = assets::load_ktx2_async(
        context.clone(),
        false,
        &Rc::new(world.ibl_lut.clone()),
        |_| {},
    )
    .await
    .map_err(|err| err.to_string())?;

    let diffuse_ibl_cubemap =
        assets::load_ktx2_cubemap(context.clone(), &Rc::new(world.diffuse_ibl_cubemap.clone()))
            .await
            .map_err(|err| err.to_string())?;

    let specular_ibl_cubemap = assets::load_ktx2_cubemap(
        context.clone(),
        &Rc::new(world.specular_ibl_cubemap.clone()),
    )
    .await
    .map_err(|err| err.to_string())?;

    let send_fn: js_sys::Function =
        js_sys::Reflect::get(&web_sys::window().unwrap(), &"send_xr_data".into())?.into();

    let player_states = Rc::new(RefCell::new(HashMap::new()));
    let movement = Rc::new(RefCell::new(None));

    setup_callbacks(
        &xr_session,
        &reference_space,
        player_states.clone(),
        movement.clone(),
        models.clone(),
        urls_to_model_slots.clone(),
        href.clone(),
        context.clone(),
    )?;

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform buffer"),
        size: std::mem::size_of::<<shared_structs::Uniforms as AsStd140>::Output>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let skybox_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("skybox uniform buffer"),
        size: std::mem::size_of::<shared_structs::SkyboxUniforms>() as u64,
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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&ibl_lut.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&diffuse_ibl_cubemap.view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&specular_ibl_cubemap.view),
            },
        ],
    });

    let skybox_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("skybox uniform bind group"),
        layout: &skybox_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: skybox_uniform_buffer.as_entire_binding(),
        }],
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

    let framebuffer_cache = ResourceCache::default();
    let bind_group_cache = ResourceCache::default();

    let mut offset = Vec3::ZERO;
    let mut orientation = glam::Quat::IDENTITY;

    let egui_ctx = egui::Context::default();
    let mut egui_renderer =
        egui_wgpu::renderer::RenderPass::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);

    let mut instance_buffer =
        InstanceBuffer::new(10, &device, wgpu::BufferUsages::VERTEX, "instance buffer");

    run_rendering_loop(&xr_session, move |time, frame| {
        let time = time / 1000.0;

        let egui_input = egui::RawInput {
            time: Some(time),
            ..Default::default()
        };

        let egui_output = egui_ctx.run(egui_input, |ctx| {
            egui::containers::Window::new("This is a window").show(ctx, |ui| {
                ui.label("This is a label");
            });
        });

        for (id, image_delta) in &egui_output.textures_delta.set {
            egui_renderer.update_texture(&device, &queue, *id, image_delta);
        }
        for id in &egui_output.textures_delta.free {
            egui_renderer.free_texture(id);
        }

        let screen_descriptor = egui_wgpu::renderer::ScreenDescriptor {
            size_in_pixels: [1024, 1024],
            pixels_per_point: 2.0,
        };

        let egui_primitives = egui_ctx.tessellate(egui_output.shapes);
        egui_renderer.update_buffers(&device, &queue, &egui_primitives, &screen_descriptor);

        if let Some(movement) = movement.borrow().as_ref() {
            offset += *movement * 2.0 / 60.0;
        }

        let reference_space = reference_space.get_offset_reference_space(
            &web_sys::XrRigidTransform::new_with_position_and_orientation(
                &vec_to_dom_point(-orientation * offset),
                &quat_to_dom_point(-orientation),
            )
            .unwrap(),
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
                }
            }

            if let Some(gamepad) = input_source.gamepad() {
                let axes = gamepad.axes();

                if let Some((x, y)) = axes
                    .get(2)
                    .as_f64()
                    .and_then(|x| axes.get(3).as_f64().map(|y| (x, y)))
                {
                    if i == 0 {
                        *movement.borrow_mut() =
                            Some(player_state.head.rotation * Vec3::new(-x as f32, 0.0, -y as f32));
                    } else if i == 1 {
                        orientation *= glam::Quat::from_rotation_y(x as f32 * 0.1 * (2.0 / 3.0));
                    }
                }
            }
        }

        let player_heads: Vec<Instance> = player_states
            .borrow()
            .values()
            .cloned()
            .map(|state| state.head)
            .collect();

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

        let player_hands: Vec<Instance> = std::iter::once(player_state.clone())
            .chain(player_states.borrow().values().cloned())
            .flat_map(|state| state.hands)
            .collect();

        let views: Vec<web_sys::XrView> = pose.views().iter().map(|view| view.into()).collect();

        let base_layer = xr_session.render_state().base_layer().unwrap();

        {
            let parse_matrix = |vec| Mat4::from_cols_array(&<[f32; 16]>::try_from(vec).unwrap());

            let left_proj = parse_matrix(views[0].projection_matrix());
            let left_inv = parse_matrix(views[0].transform().matrix()).inverse();

            let left_projection_view = (left_proj * left_inv).into();
            let left_instance = Instance::from_transform(views[0].transform(), 0.0);

            let (right_projection_view, right_proj, right_instance) =
                if let Some(right_view) = views.get(1) {
                    let right_inv = parse_matrix(right_view.transform().matrix()).inverse();
                    let right_proj = parse_matrix(right_view.projection_matrix());

                    (
                        (right_proj * right_inv).into(),
                        right_proj,
                        Instance::from_transform(right_view.transform(), 0.0),
                    )
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
                        left_eye_position: left_instance.position,
                        right_eye_position: right_instance.position,
                        render_direct_to_framebuffer: render_direct_to_framebuffer as u32,
                        inline_tonemapping: inline_tonemapping as u32,
                    }
                    .as_std140(),
                ),
            );

            queue.write_buffer(
                &skybox_uniform_buffer,
                0,
                bytemuck::bytes_of(
                    &shared_structs::SkyboxUniforms {
                        left_projection_inverse: left_proj.inverse().into(),
                        right_projection_inverse: right_proj.inverse().into(),
                        left_view_inverse: left_instance.rotation.into(),
                        right_view_inverse: right_instance.rotation.into(),
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
        }

        let framebuffer: web_sys::WebGlFramebuffer =
            js_sys::Reflect::get(&base_layer, &"framebuffer".into())
                .unwrap()
                .into();

        let framebuffer_colour_attachment = create_view_from_device_framebuffer(
            &device,
            framebuffer.clone(),
            &base_layer,
            wgpu::TextureFormat::Rgba8Unorm,
            "device framebuffer (colour)",
        );

        let num_views = multiview.map(|views| views.get()).unwrap_or(1);

        // todo: resize this if the frmaebuffer dimensions change.
        let hdr_or_multiview_framebuffer = framebuffer_cache.get("hdr framebuffer", || {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("hdr framebuffer"),
                size: wgpu::Extent3d {
                    width: base_layer.framebuffer_width() / num_views,
                    height: base_layer.framebuffer_height(),
                    depth_or_array_layers: num_views,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: if inline_tonemapping {
                    wgpu::TextureFormat::Rgba8Unorm
                } else {
                    wgpu::TextureFormat::Rgba16Float
                },
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC,
            });

            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(if multiview.is_none() {
                    wgpu::TextureViewDimension::D2
                } else {
                    wgpu::TextureViewDimension::D2Array
                }),
                ..Default::default()
            });

            renderer_core::Texture { texture, view }
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
                        resource: wgpu::BindingResource::TextureView(
                            &hdr_or_multiview_framebuffer.view,
                        ),
                    },
                ],
            })
        });

        let depth = if render_direct_to_framebuffer {
            BorrowedOrOwnedFramebuffer::Owned(create_view_from_device_framebuffer(
                &device,
                framebuffer,
                &base_layer,
                wgpu::TextureFormat::Depth24PlusStencil8,
                "device framebuffer (depth)",
            ))
        } else {
            // todo: resize this if the frmaebuffer dimensions change.
            BorrowedOrOwnedFramebuffer::Borrowed(framebuffer_cache.get("depth", || {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
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
                });

                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(if multiview.is_none() {
                        wgpu::TextureViewDimension::D2
                    } else {
                        wgpu::TextureViewDimension::D2Array
                    }),
                    ..Default::default()
                });

                renderer_core::Texture { texture, view }
            }))
        };

        let ui_texture = framebuffer_cache.get("ui texture", || {
            renderer_core::Texture::new(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("ui texture"),
                size: wgpu::Extent3d {
                    width: 1024,
                    height: 1024,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
            }))
        });

        let ui_texture_bind_group = bind_group_cache.get("ui texture bind group", || {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ui texture bind group"),
                layout: &ui_texture_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&ui_texture.view),
                }],
            })
        });

        // We borrow all the bind groups here because they need to be borrowed for the entire duration of the render pass.

        let model_bind_groups = ModelBindGroups {
            opaque: models
                .values()
                .map(|model| {
                    model
                        .model
                        .iter()
                        .flat_map(|model| &model.opaque_primitives)
                        .map(|primitive| primitive.bind_group.borrow())
                        .collect::<Vec<_>>()
                })
                .collect(),
            alpha_clipped: models
                .values()
                .map(|model| {
                    model
                        .model
                        .iter()
                        .flat_map(|model| &model.alpha_clipped_primitives)
                        .map(|primitive| primitive.bind_group.borrow())
                        .collect::<Vec<_>>()
                })
                .collect(),
            opaque_double_sided: models
                .values()
                .map(|model| {
                    model
                        .model
                        .iter()
                        .flat_map(|model| &model.opaque_double_sided_primitives)
                        .map(|primitive| primitive.bind_group.borrow())
                        .collect::<Vec<_>>()
                })
                .collect(),
            alpha_clipped_double_sided: models
                .values()
                .map(|model| {
                    model
                        .model
                        .iter()
                        .flat_map(|model| &model.alpha_clipped_double_sided_primitives)
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

        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("command encoder"),
        });

        instance_buffer.clear();

        let model_instances: Vec<_> = models
            .values()
            .map(|model| {
                instance_buffer.push(&model.instances, &device, &queue, &mut command_encoder)
            })
            .collect();

        let heads = HeadOrHandsRenderingData {
            instance_range: instance_buffer.push(
                &player_heads,
                &device,
                &queue,
                &mut command_encoder,
            ),
            model: &head_model,
            bind_groups: &head_primitives,
        };

        let heads_mirrored = HeadOrHandsRenderingData {
            instance_range: instance_buffer.push(
                &player_heads_mirrored,
                &device,
                &queue,
                &mut command_encoder,
            ),
            model: &head_model,
            bind_groups: &head_primitives,
        };

        let hands = HeadOrHandsRenderingData {
            instance_range: instance_buffer.push(
                &player_hands,
                &device,
                &queue,
                &mut command_encoder,
            ),
            model: &hand_model,
            bind_groups: &hand_primitives,
        };

        let mirror_model = mirror_model.as_ref().map(|mirror_model| {
            (
                mirror_model,
                instance_buffer.push(
                    &mirror_model.instances,
                    &device,
                    &queue,
                    &mut command_encoder,
                ),
                &mirror_uniform_bind_group,
            )
        });

        let ui_plane_instance_range = instance_buffer.push(
            &ui_plane_model.instances,
            &device,
            &queue,
            &mut command_encoder,
        );

        egui_renderer.execute(
            &mut command_encoder,
            &ui_texture.view,
            &egui_primitives,
            &screen_descriptor,
            Some(wgpu::Color::TRANSPARENT),
        );

        let vertex_buffers = vertex_buffers.borrow();
        let index_buffer = index_buffer.borrow();

        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main render pass"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: if render_direct_to_framebuffer {
                    &framebuffer_colour_attachment.view
                } else {
                    &hdr_or_multiview_framebuffer.view
                },
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth.get().view,
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

        render_everything(
            &mut render_pass,
            &pipelines,
            &models,
            &model_instances,
            &model_bind_groups,
            &index_buffer,
            &vertex_buffers,
            &instance_buffer,
            &uniform_bind_group,
            &hands,
            &heads,
            &heads_mirrored,
            mirror_model,
            &UiData {
                plane_instance_range: ui_plane_instance_range,
                plane_model: ui_plane_model.model.as_ref().unwrap(),
                texture_bind_group: ui_texture_bind_group,
            },
            &skybox_uniform_bind_group,
            render_skybox,
        );

        drop(render_pass);

        if !render_direct_to_framebuffer {
            // Blit from the intermediate framebuffer to the device framebuffer,
            // either tonemapping the colour or just blitting.
            //
            // Todo: it'd be great to be able to just do a blit here if `inline_tonemapping` is true.
            // Doing this through wgpu would involve a lot of code changes to glow etc. but we could just use raw gl potentially.

            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tonemap render pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &framebuffer_colour_attachment.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&pipelines.tonemap);

            render_pass.set_bind_group(0, &uniform_bind_group, &[]);
            render_pass.set_bind_group(1, tonemap_bind_group, &[]);

            render_pass.draw(0..3, 0..1);

            drop(render_pass);
        }

        queue.submit(std::iter::once(command_encoder.finish()));
    });

    Ok(())
}

fn setup_callbacks(
    xr_session: &web_sys::XrSession,
    reference_space: &web_sys::XrReferenceSpace,
    player_states: Rc<RefCell<HashMap<String, PlayerState>>>,
    movement: Rc<RefCell<Option<Vec3>>>,
    models: Rc<RefCell<slotmap::SlotMap<slotmap::DefaultKey, InstancedModel>>>,
    urls_to_model_slots: Rc<RefCell<HashMap<String, slotmap::DefaultKey>>>,
    href: url::Url,
    context: Rc<ModelLoadContext>,
) -> Result<(), wasm_bindgen::JsValue> {
    let setup_fn: js_sys::Function =
        js_sys::Reflect::get(&web_sys::window().unwrap(), &"set_xr_data_handler".into())?.into();

    let on_message = {
        wasm_bindgen::closure::Closure::wrap(Box::new({
            let player_states = Rc::clone(&player_states);

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
            }
        })
            as Box<dyn FnMut(js_sys::Uint8Array, String)>)
    };

    setup_fn.call1(
        &wasm_bindgen::JsValue::undefined(),
        on_message.as_ref().unchecked_ref(),
    )?;
    // We need do this this as otherwise `on_message` is dropped when `run()` finishes.
    on_message.forget();

    let spawn_setup_fn: js_sys::Function =
        js_sys::Reflect::get(&web_sys::window().unwrap(), &"set_handle_spawn".into())?.into();

    let handle_spawn = wasm_bindgen::closure::Closure::wrap(Box::new({
        let handle_spawn_fallible =
            move |url: String, position: js_sys::Array| -> anyhow::Result<()> {
                let position = glam::DVec3::new(
                    position
                        .get(0)
                        .as_f64()
                        .ok_or_else(|| anyhow::anyhow!("Failed to parse position"))?,
                    position
                        .get(1)
                        .as_f64()
                        .ok_or_else(|| anyhow::anyhow!("Failed to parse position"))?,
                    position
                        .get(2)
                        .as_f64()
                        .ok_or_else(|| anyhow::anyhow!("Failed to parse position"))?,
                )
                .as_vec3();

                let instance = Instance::new(position, 1.0, Default::default());

                let (slot, new_model) = match urls_to_model_slots.borrow().get(&url) {
                    Some(slot) => (*slot, false),
                    None => {
                        let instances = vec![instance];

                        let slot = models.borrow_mut().insert(InstancedModel {
                            model: None,
                            instances,
                        });

                        (slot, true)
                    }
                };

                if !new_model {
                    if let Some(instanced_model) = models.borrow_mut().get_mut(slot) {
                        instanced_model.instances.push(instance);
                    } else {
                        log::warn!(
                            "Spawn failed: no model found for slot {:?}, url '{}'",
                            slot,
                            url
                        );
                    }
                } else {
                    let model_url = url::Url::options().base_url(Some(&href)).parse(&url)?;

                    urls_to_model_slots
                        .borrow_mut()
                        .insert(model_url.as_str().to_string(), slot);

                    let context = context.clone();
                    let models = models.clone();

                    wasm_bindgen_futures::spawn_local(async move {
                        if let Err(error) = load_model(model_url, slot, context, &models).await {
                            log::error!("Failed to load model: {}", error);
                        }
                    });
                }

                Ok(())
            };

        move |url: String, position: js_sys::Array| {
            if let Err(error) = handle_spawn_fallible(url, position) {
                log::error!("Failed to spawn: {}", error);
            }
        }
    })
        as Box<dyn FnMut(String, js_sys::Array)>);

    spawn_setup_fn.call1(
        &wasm_bindgen::JsValue::undefined(),
        handle_spawn.as_ref().unchecked_ref(),
    )?;

    handle_spawn.forget();

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

    Ok(())
}

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct Instance {
    pub(crate) position: Vec3,
    pub(crate) scale: f32,
    pub(crate) rotation: glam::Quat,
}

impl Instance {
    pub(crate) fn new(position: Vec3, scale: f32, rotation: glam::Quat) -> Self {
        Self {
            position,
            scale,
            rotation,
        }
    }

    pub(crate) fn from_transform(transform: web_sys::XrRigidTransform, scale: f32) -> Self {
        let rotation = transform.orientation();

        let rotation =
            glam::DQuat::from_xyzw(rotation.x(), rotation.y(), rotation.z(), rotation.w());
        Self {
            position: transform_to_position_vec3(&transform),
            rotation: rotation.as_f32(),
            scale,
        }
    }
}

fn transform_to_position_vec3(transform: &web_sys::XrRigidTransform) -> Vec3 {
    let position = transform.position();
    let position = glam::DVec3::new(position.x(), position.y(), position.z());
    position.as_vec3()
}

impl Default for Instance {
    fn default() -> Self {
        Self::new(Vec3::ZERO, 1.0, glam::Quat::IDENTITY)
    }
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

struct ModelBindGroups<'a> {
    opaque: Vec<Vec<std::cell::Ref<'a, wgpu::BindGroup>>>,
    alpha_clipped: Vec<Vec<std::cell::Ref<'a, wgpu::BindGroup>>>,
    opaque_double_sided: Vec<Vec<std::cell::Ref<'a, wgpu::BindGroup>>>,
    alpha_clipped_double_sided: Vec<Vec<std::cell::Ref<'a, wgpu::BindGroup>>>,
}

struct HeadOrHandsRenderingData<'a> {
    instance_range: Range<u32>,
    model: &'a assets::Model,
    bind_groups: &'a [std::cell::Ref<'a, wgpu::BindGroup>],
}

fn render_all_model_primitives<'a, F: Fn(&'a assets::Model) -> &'a [ModelPrimitive]>(
    render_pass: &mut wgpu::RenderPass<'a>,
    models: &'a slotmap::SlotMap<slotmap::DefaultKey, InstancedModel>,
    model_instances: &[Range<u32>],
    primitives_getter: F,
    bind_groups: &'a [Vec<std::cell::Ref<wgpu::BindGroup>>],
    heads: &HeadOrHandsRenderingData<'a>,
    hands: &HeadOrHandsRenderingData<'a>,
) {
    for (model_index, (instanced_model, instance_range)) in
        models.values().zip(model_instances).enumerate()
    {
        if let Some(model) = &instanced_model.model {
            for (primitive_index, primitive) in primitives_getter(model).iter().enumerate() {
                render_pass.set_bind_group(1, &bind_groups[model_index][primitive_index], &[]);
                render_pass.draw_indexed(
                    primitive.index_buffer_range.clone(),
                    0,
                    instance_range.clone(),
                );
            }
        }
    }

    for (primitive_index, primitive) in primitives_getter(heads.model).iter().enumerate() {
        render_pass.set_bind_group(1, &heads.bind_groups[primitive_index], &[]);
        render_pass.draw_indexed(
            primitive.index_buffer_range.clone(),
            0,
            heads.instance_range.clone(),
        );
    }

    for (primitive_index, primitive) in primitives_getter(hands.model).iter().enumerate() {
        render_pass.set_bind_group(1, &hands.bind_groups[primitive_index], &[]);
        render_pass.draw_indexed(
            primitive.index_buffer_range.clone(),
            0,
            hands.instance_range.clone(),
        );
    }
}

struct UiData<'a> {
    plane_instance_range: Range<u32>,
    plane_model: &'a assets::Model,
    texture_bind_group: &'a wgpu::BindGroup,
}

fn render_everything<'a>(
    render_pass: &mut wgpu::RenderPass<'a>,
    pipelines: &'a Pipelines,
    models: &'a slotmap::SlotMap<slotmap::DefaultKey, InstancedModel>,
    model_instances: &[Range<u32>],
    model_bind_groups: &'a ModelBindGroups,
    index_buffer: &'a IndexBuffer,
    vertex_buffers: &'a VertexBuffers,
    instance_buffer: &'a InstanceBuffer,
    uniform_bind_group: &'a wgpu::BindGroup,
    hands: &'a HeadOrHandsRenderingData,
    heads: &'a HeadOrHandsRenderingData,
    heads_mirrored: &'a HeadOrHandsRenderingData,
    mirror_model: Option<(&InstancedModel, Range<u32>, &'a wgpu::BindGroup)>,
    ui_data: &UiData<'a>,
    skybox_uniform_bind_group: &'a wgpu::BindGroup,
    render_skybox: bool,
) {
    render_pass.set_bind_group(0, uniform_bind_group, &[]);

    render_pass.set_index_buffer(index_buffer.buffer.slice(..), wgpu::IndexFormat::Uint32);
    render_pass.set_vertex_buffer(0, vertex_buffers.position.slice(..));
    render_pass.set_vertex_buffer(1, vertex_buffers.normal.slice(..));
    render_pass.set_vertex_buffer(2, vertex_buffers.uv.slice(..));
    render_pass.set_vertex_buffer(3, instance_buffer.buffer.slice(..));

    if let Some((mirror_model, mirror_model_instances, mirror_uniform_bind_group)) = mirror_model {
        let mirror_model = mirror_model.model.as_ref().unwrap();

        render_pass.set_stencil_reference(1);

        render_pass.set_pipeline(&pipelines.stencil_write);

        render_pass.draw_indexed(
            mirror_model.index_buffer_range.clone(),
            0,
            mirror_model_instances.clone(),
        );

        render_pass.set_bind_group(2, mirror_uniform_bind_group, &[]);

        {
            render_pass.set_pipeline(&pipelines.pbr.opaque_mirrored);

            render_all_model_primitives(
                render_pass,
                models,
                model_instances,
                |model| &model.opaque_primitives,
                &model_bind_groups.opaque,
                heads_mirrored,
                hands,
            );

            render_pass.set_pipeline(&pipelines.pbr_double_sided.opaque_mirrored);

            render_all_model_primitives(
                render_pass,
                models,
                model_instances,
                |model| &model.opaque_double_sided_primitives,
                &model_bind_groups.opaque_double_sided,
                heads_mirrored,
                hands,
            );

            render_pass.set_pipeline(&pipelines.pbr.alpha_clipped_mirrored);

            render_all_model_primitives(
                render_pass,
                models,
                model_instances,
                |model| &model.alpha_clipped_primitives,
                &model_bind_groups.alpha_clipped,
                heads_mirrored,
                hands,
            );

            render_pass.set_pipeline(&pipelines.pbr_double_sided.alpha_clipped_mirrored);

            render_all_model_primitives(
                render_pass,
                models,
                model_instances,
                |model| &model.alpha_clipped_double_sided_primitives,
                &model_bind_groups.alpha_clipped_double_sided,
                heads_mirrored,
                hands,
            );
        }

        if render_skybox {
            render_pass.set_pipeline(&pipelines.skybox_mirrored);
            render_pass.set_bind_group(1, skybox_uniform_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        {
            render_pass.set_pipeline(&pipelines.set_depth);
            render_pass.draw_indexed(
                mirror_model.index_buffer_range.clone(),
                0,
                mirror_model_instances,
            );
        }
    }

    {
        render_pass.set_pipeline(&pipelines.pbr.opaque);

        render_all_model_primitives(
            render_pass,
            models,
            model_instances,
            |model| &model.opaque_primitives,
            &model_bind_groups.opaque,
            heads,
            hands,
        );

        render_pass.set_pipeline(&pipelines.pbr_double_sided.opaque);

        render_all_model_primitives(
            render_pass,
            models,
            model_instances,
            |model| &model.opaque_double_sided_primitives,
            &model_bind_groups.opaque_double_sided,
            heads,
            hands,
        );

        render_pass.set_pipeline(&pipelines.pbr.alpha_clipped);

        render_all_model_primitives(
            render_pass,
            models,
            model_instances,
            |model| &model.alpha_clipped_primitives,
            &model_bind_groups.alpha_clipped,
            heads,
            hands,
        );

        render_pass.set_pipeline(&pipelines.pbr_double_sided.alpha_clipped);

        render_all_model_primitives(
            render_pass,
            models,
            model_instances,
            |model| &model.alpha_clipped_double_sided_primitives,
            &model_bind_groups.alpha_clipped_double_sided,
            heads,
            hands,
        );
    }

    if render_skybox {
        render_pass.set_pipeline(&pipelines.skybox);
        render_pass.set_bind_group(1, skybox_uniform_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    {
        render_pass.set_pipeline(&pipelines.ui);

        render_pass.set_bind_group(1, ui_data.texture_bind_group, &[]);
        render_pass.draw_indexed(
            ui_data.plane_model.index_buffer_range.clone(),
            0,
            ui_data.plane_instance_range.clone(),
        );
    }
}

struct InstancedModel {
    model: Option<assets::Model>,
    instances: Vec<Instance>,
}

#[derive(serde::Deserialize)]
struct World {
    models: Vec<ModelReference>,
    mirror: Option<ModelReference>,
    #[serde(deserialize_with = "deserialize_relative_url")]
    ibl_lut: url::Url,
    #[serde(deserialize_with = "deserialize_relative_url")]
    diffuse_ibl_cubemap: url::Url,
    #[serde(deserialize_with = "deserialize_relative_url")]
    specular_ibl_cubemap: url::Url,
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

type Models = Rc<RefCell<slotmap::SlotMap<slotmap::DefaultKey, InstancedModel>>>;

async fn load_model(
    url: url::Url,
    slot: slotmap::DefaultKey,
    context: Rc<ModelLoadContext>,
    models: &Models,
) -> anyhow::Result<()> {
    let bytes = context.request_client.fetch_bytes(&url, None).await?;
    let model = load_gltf_from_bytes(&bytes, Some(url), &context).await?;

    if let Some(instanced_model) = models.borrow_mut().get_mut(slot) {
        instanced_model.model = Some(model);
    }

    Ok(())
}

fn vec_to_dom_point(vec: Vec3) -> web_sys::DomPointInit {
    let mut dom_point = web_sys::DomPointInit::new();

    dom_point.x(vec.x as f64).y(vec.y as f64).z(vec.z as f64);

    dom_point
}

fn quat_to_dom_point(quat: glam::Quat) -> web_sys::DomPointInit {
    let mut dom_point = web_sys::DomPointInit::new();

    dom_point
        .x(quat.x as f64)
        .y(quat.y as f64)
        .z(quat.z as f64)
        .w(quat.w as f64);

    dom_point
}

enum BorrowedOrOwnedFramebuffer<'a> {
    Owned(renderer_core::Texture),
    Borrowed(&'a renderer_core::Texture),
}

impl<'a> BorrowedOrOwnedFramebuffer<'a> {
    fn get(&'a self) -> &'a renderer_core::Texture {
        match self {
            Self::Owned(texture) => texture,
            Self::Borrowed(texture) => texture,
        }
    }
}
