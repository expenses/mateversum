use bevy_app::{App, Plugin};
use bevy_ecs::prelude::{Entity, SystemStage};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

pub mod components;
mod systems;

pub use bevy_app;
pub use bevy_ecs;
pub use renderer_core;
pub use url;

pub use renderer_core::{assets::textures, glam::Vec3, utils::Swappable};

use components::Instance;

pub struct Device(Arc<wgpu::Device>);
pub struct Queue(Arc<wgpu::Queue>);
struct Pipelines(Arc<renderer_core::Pipelines>);
struct BindGroupLayouts(Arc<renderer_core::BindGroupLayouts>);

struct UniformBuffer(Arc<wgpu::Buffer>);
struct MainBindGroup(Swappable<wgpu::BindGroup>);
struct SkyboxUniformBuffer(wgpu::Buffer);
struct SkyboxUniformBindGroup(wgpu::BindGroup);

struct IndexBuffer(Arc<parking_lot::Mutex<renderer_core::IndexBuffer>>);
struct VertexBuffers(Arc<parking_lot::Mutex<renderer_core::VertexBuffers>>);
struct InstanceBuffer(renderer_core::InstanceBuffer);

pub struct FrameTime(pub f64);

struct IntermediateDepthFramebuffer(Option<renderer_core::Texture>);
struct IntermediateColorFramebuffer(Option<renderer_core::Texture>);
struct CompositeBindGroup(Option<wgpu::BindGroup>);
struct LinearSampler(Arc<wgpu::Sampler>);

struct ModelUrls(pub HashMap<url::Url, Entity>);

pub struct NewIblTextures(pub Option<NewIblTexturesInner>);

pub struct NewIblTexturesInner {
    pub diffuse_cubemap: url::Url,
    pub specular_cubemap: url::Url,
}

#[derive(bevy_ecs::prelude::StageLabel, Debug, PartialEq, Eq, Clone, Hash)]
pub enum StartupStage {
    PipelineCreation,
    BindGroupCreation,
}

#[derive(Default)]
pub struct XrPlugin;

impl Plugin for XrPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ModelUrls(Default::default()));
        app.insert_resource(textures::Settings {
            anisotropy_clamp: Some(std::num::NonZeroU8::new(16).unwrap()),
        });
        app.insert_resource(NewIblTextures(None));

        app.add_startup_stage(
            StartupStage::PipelineCreation,
            SystemStage::single_threaded()
                .with_system(systems::create_bind_group_layouts_and_pipelines),
        );
        app.add_startup_stage_after(
            StartupStage::PipelineCreation,
            StartupStage::BindGroupCreation,
            SystemStage::single_threaded().with_system(systems::allocate_bind_groups),
        );

        app.add_system(systems::start_loading_models);
        app.add_system(systems::finish_loading_models);
        app.add_system(systems::update_ibl_textures);

        app.add_system(systems::update_uniform_buffers);
        app.add_system(systems::clear_instance_buffer);
        app.add_system(systems::push_entity_instances);
        app.add_system(systems::upload_instances);
        app.add_system(systems::rendering::render);

        app.world
            .spawn()
            .insert(Instance(renderer_core::Instance::new(
                Vec3::new(1.0, 1.0, -2.0),
                1.0,
                Default::default(),
            )));

        app.world
            .spawn()
            .insert(Instance(renderer_core::Instance::new(
                Vec3::new(-1.0, 1.0, -2.0),
                1.0,
                Default::default(),
            )));
    }
}

pub enum Mode {
    Vr,
    Ar,
    Desktop,
}

pub async fn initialise(
    mode: Mode,
) -> (
    web_sys::XrSession,
    web_sys::XrReferenceSpace,
    Device,
    Queue,
    renderer_core::PipelineOptions,
) {
    let mode = match mode {
        Mode::Vr => web_sys::XrSessionMode::ImmersiveVr,
        Mode::Ar => web_sys::XrSessionMode::ImmersiveAr,
        Mode::Desktop => unimplemented!("A desktop mode isn't supported yet"),
    };

    let navigator = web_sys::window().unwrap().navigator();
    let xr = navigator.xr();

    let required_features = js_sys::Array::of1(&"local-floor".into());

    let xr_session: web_sys::XrSession =
        wasm_bindgen_futures::JsFuture::from(xr.request_session_with_options(
            mode,
            web_sys::XrSessionInit::new().required_features(&required_features),
        ))
        .await
        .unwrap()
        .into();

    let canvas = renderer_core::Canvas::default();

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

    let mut layer_init = web_sys::XrWebGlLayerInit::new();

    let pipeline_options = if mode == web_sys::XrSessionMode::ImmersiveVr {
        renderer_core::PipelineOptions {
            multiview: Some(std::num::NonZeroU32::new(2).unwrap()),
            inline_tonemapping: true,
        }
    } else {
        renderer_core::PipelineOptions {
            multiview: None,
            inline_tonemapping: true,
        }
    };

    layer_init
        .depth(pipeline_options.render_direct_to_framebuffer())
        .stencil(pipeline_options.render_direct_to_framebuffer());

    let webgl2_context =
        canvas.create_webgl2_context(renderer_core::ContextCreationOptions { stencil: true });

    wasm_bindgen_futures::JsFuture::from(webgl2_context.make_xr_compatible())
        .await
        .expect("Failed to make the webgl context xr-compatible");

    let xr_gl_layer = web_sys::XrWebGlLayer::new_with_web_gl2_rendering_context_and_layer_init(
        &xr_session,
        &webgl2_context,
        &layer_init,
    )
    .unwrap();

    let mut render_state_init = web_sys::XrRenderStateInit::new();
    render_state_init
        .depth_near(0.001)
        .base_layer(Some(&xr_gl_layer));
    xr_session.update_render_state_with_state(&render_state_init);

    let reference_space_type = match mode {
        web_sys::XrSessionMode::Inline => web_sys::XrReferenceSpaceType::Viewer,
        _ => web_sys::XrReferenceSpaceType::LocalFloor,
    };

    let xr_reference_space: web_sys::XrReferenceSpace = wasm_bindgen_futures::JsFuture::from(
        xr_session.request_reference_space(reference_space_type),
    )
    .await
    .unwrap()
    .into();

    (
        xr_session,
        xr_reference_space,
        Device(Arc::new(device)),
        Queue(Arc::new(queue)),
        pipeline_options,
    )
}

#[derive(Clone)]
struct SimpleHttpClient;

impl renderer_core::assets::HttpClient for SimpleHttpClient {
    type Future = std::pin::Pin<Box<dyn core::future::Future<Output = anyhow::Result<Vec<u8>>>>>;

    fn fetch_bytes(&self, url: &url::Url, byte_range: Option<Range<usize>>) -> Self::Future {
        async fn resolve_promise(
            promise: js_sys::Promise,
        ) -> anyhow::Result<wasm_bindgen::JsValue> {
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|err| anyhow::anyhow!("{:?}", err))
        }

        let url = url.clone();

        Box::pin(async move {
            let request_init = construct_request_init(byte_range.clone())?;

            let request = web_sys::Request::new_with_str_and_init(url.as_str(), &request_init)
                .map_err(|err| anyhow::anyhow!("{:?}", err))?;

            let response: web_sys::Response =
                resolve_promise(web_sys::window().unwrap().fetch_with_request(&request))
                    .await?
                    .into();

            let array_buffer: js_sys::ArrayBuffer = resolve_promise(
                response
                    .array_buffer()
                    .map_err(|err| anyhow::anyhow!("{:?}", err))?,
            )
            .await?
            .into();

            let uint8_array = js_sys::Uint8Array::new(&array_buffer);

            Ok(uint8_array.to_vec())
        })
    }
}

fn construct_request_init(
    byte_range: Option<Range<usize>>,
) -> anyhow::Result<web_sys::RequestInit> {
    let mut request_init = web_sys::RequestInit::new();

    fn byte_range_string(range: Range<usize>) -> String {
        format!("bytes={}-{}", range.start, range.end - 1)
    }

    if let Some(byte_range) = byte_range {
        let headers = js_sys::Object::new();
        js_sys::Reflect::set(
            &headers,
            &"Range".into(),
            &byte_range_string(byte_range).into(),
        )
        .map_err(|err| anyhow::anyhow!("Js Error: {:?}", err))?;
        request_init.headers(&headers);
    }

    Ok(request_init)
}
