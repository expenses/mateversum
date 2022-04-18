use futures::FutureExt;
use glam::{Mat4, Vec3};
use kiss_engine_wgpu::{
    BindingResource, Device, RenderPipelineDesc, Resource, ShaderSettings, Texture,
    VertexBufferLayout,
};
use wasm_webxr_helpers::{button_click_future, create_button};
use wgpu::util::DeviceExt;

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
    render_state_init.base_layer(Some(&xr_gl_layer));
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

    let mut device: kiss_engine_wgpu::Device<(&str, u32, u32)> = kiss_engine_wgpu::Device::new(device);

    let mut images = Images::default();

    let mut context = ModelLoadContext {
        black_image_id: images.push(load_image(
            &device,
            &queue,
            1,
            1,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &[0, 0, 0, 255],
        )),
        default_metallic_roughness_image_id: images.push(load_image(
            &device,
            &queue,
            1,
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            &[0, 255, 0, 255],
        )),
        flat_normals_image_id: images.push(load_image(
            &device,
            &queue,
            1,
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            &[0, 255, 0, 255],
        )),
        white_image_id: images.push(load_image(
            &device,
            &queue,
            1,
            1,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &[255, 255, 255, 255],
        )),
        url: url::Url::parse(
            "http://localhost:8000/sample_models/Sponza/glTF/Sponza.gltf",
        )
        .unwrap(),
        device: &device,
        queue: &queue,
        images: &mut images,
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

    let linear_sampler =
        device.create_resource(device.inner.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));

    wasm_webxr_helpers::Session { inner: xr_session }.run_rendering_loop(move |time, frame| {
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
            let base_transform = Mat4::from_scale(Vec3::splat(0.25));
            let parse_matrix = |vec| Mat4::from_cols_array(&<[f32; 16]>::try_from(vec).unwrap());

            let left_proj = parse_matrix(views[0].projection_matrix());
            let left_inv = parse_matrix(views[0].transform().inverse().matrix());

            queue.write_buffer(
                &left_eye_uniform_buffer,
                0,
                bytemuck::bytes_of(&shared_structs::Uniforms {
                    projection_view: { left_proj * (left_inv * base_transform) }.into(),
                    eye_position: {
                        let p = views[0].transform().position();
                        glam::DVec3::new(p.x(), p.y(), p.z()).as_vec3()
                    },
                    _padding: 0,
                    ..Default::default()
                }),
            );

            if let Some(right_view) = views.get(1) {
                let right_inv = parse_matrix(right_view.transform().inverse().matrix());
                let right_proj = parse_matrix(right_view.projection_matrix());

                queue.write_buffer(
                    &right_eye_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&shared_structs::Uniforms {
                        projection_view: { right_proj * (right_inv * base_transform) }.into(),
                        eye_position: {
                            let p = right_view.transform().position();
                            glam::DVec3::new(p.x(), p.y(), p.z()).as_vec3()
                        },
                        _padding: 0,
                        ..Default::default()
                    }),
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

            let pipeline = device.get_pipeline(
                "pbr pipeline",
                device.device.get_shader(
                    "vertex.spv",
                    include_bytes!("../vertex.spv"),
                    ShaderSettings {
                        entry_point: "vertex",
                        ..Default::default()
                    },
                ),
                device.device.get_shader(
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
                ],
            );

            render_pass.set_pipeline(&pipeline.pipeline);

            for (primitive_id, primitive) in model.primitives.iter().enumerate() {
                render_pass.set_vertex_buffer(0, primitive.positions.slice(..));
                render_pass.set_vertex_buffer(1, primitive.normals.slice(..));
                render_pass.set_vertex_buffer(2, primitive.uvs.slice(..));
                render_pass
                    .set_index_buffer(primitive.indices.slice(..), wgpu::IndexFormat::Uint32);

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
                        ("pbr pipeline", i as u32, primitive_id as u32),
                        pipeline,
                        &[
                            uniform_buffer(i),
                            BindingResource::Sampler(&linear_sampler),
                            images.get(primitive.albedo_texture_id),
                            images.get(primitive.normal_texture_id),
                            images.get(primitive.metallic_roughness_texture_id),
                            images.get(primitive.emissive_texture_id),
                        ],
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

struct ModelLoadContext<'a> {
    device: &'a Device<(&'static str, u32, u32)>,
    queue: &'a wgpu::Queue,
    url: url::Url,
    images: &'a mut Images,
    black_image_id: usize,
    white_image_id: usize,
    default_metallic_roughness_image_id: usize,
    flat_normals_image_id: usize,
}

struct ModelBuffers<'a> {
    map: std::collections::HashMap<usize, Vec<u8>>,
    blob: Option<&'a Vec<u8>>
}

struct Model {
    primitives: Vec<ModelPrimitive>,
}

async fn load_gltf(context: &mut ModelLoadContext<'_>) -> Model {
    let bytes = fetch_bytes(&context.url).await;

    let gltf = gltf::Gltf::from_slice(&bytes).unwrap();

    let mut buffers = ModelBuffers {
        blob: gltf.blob.as_ref(),
        map: Default::default()
    };

    let buffer_blob = gltf.blob.as_ref();

    let node_tree = gltf_helpers::NodeTree::new(gltf.nodes());

    let mut primitives = Vec::new();

    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => {}
            gltf::buffer::Source::Uri(uri) => {
                let url = url::Url::options()
                    .base_url(Some(&context.url))
                    .parse(uri)
                    .unwrap();

                if url.scheme() == "data" {
                    let (mime_type, data) = url.path().split_once(',').unwrap();
                    log::info!("Got: {}", mime_type);
                    buffers.map.insert(buffer.index(), base64::decode(data).unwrap());
                } else {
                    buffers.map.insert(buffer.index(), fetch_bytes(&url).await);
                }
            }
        }
    }

    for (node, mesh) in gltf
        .nodes()
        .filter_map(|node| node.mesh().map(|mesh| (node, mesh)))
    {
        let transform = node_tree.transform_of(node.index());

        log::info!("transform: {:?}", transform);
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| match buffer.source() {
                gltf::buffer::Source::Bin => Some(buffers.blob.unwrap()),
                gltf::buffer::Source::Uri(_) => buffers.map.get(&buffer.index()).map(|vec| &vec[..]),
            });

            let indices: Vec<_> = reader.read_indices().unwrap().into_u32().collect();

            let positions: Vec<glam::Vec3> = reader
                .read_positions()
                .unwrap()
                .map(|pos| transform * Vec3::from(pos))
                .collect();
            let normals: Vec<glam::Vec3> = reader
                .read_normals()
                .unwrap()
                .map(|rot| transform.rotation * Vec3::from(rot))
                .collect();
            let uvs: Vec<glam::Vec2> = reader
                .read_tex_coords(0)
                .unwrap()
                .into_f32()
                .map(glam::Vec2::from)
                .collect();

            let material = primitive.material();

            let pbr = material.pbr_metallic_roughness();

            log::info!(
                "{:?}",
                (
                    pbr.metallic_factor(),
                    pbr.roughness_factor(),
                    pbr.base_color_factor(),
                    material.emissive_factor(),
                )
            );

            let primitive = ModelPrimitive {
                indices: context.device.inner.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("indices"),
                        contents: bytemuck::cast_slice(&indices),
                        usage: wgpu::BufferUsages::INDEX,
                    },
                ),
                positions: context.device.inner.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("positions"),
                        contents: bytemuck::cast_slice(&positions),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                ),
                normals: context.device.inner.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("normals"),
                        contents: bytemuck::cast_slice(&normals),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                ),
                uvs: context
                    .device
                    .inner
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("uvs"),
                        contents: bytemuck::cast_slice(&uvs),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                albedo_texture_id: if let Some(albedo_texture) = pbr.base_color_texture() {
                    context.images.push(
                        load_image_from_gltf(
                            &albedo_texture.texture(),
                            true,
                            &buffers,
                            context,
                        )
                        .await,
                    )
                } else {
                    context.white_image_id
                },
                normal_texture_id: if let Some(normal_texture) = material.normal_texture() {
                    context.images.push(
                        load_image_from_gltf(
                            &normal_texture.texture(),
                            false,
                            &buffers,
                            context,
                        )
                        .await,
                    )
                } else {
                    context.flat_normals_image_id
                },
                metallic_roughness_texture_id: if let Some(metallic_roughness_texture) =
                    pbr.metallic_roughness_texture()
                {
                    context.images.push(
                        load_image_from_gltf(
                            &metallic_roughness_texture.texture(),
                            false,
                            &buffers,
                            context,
                        )
                        .await,
                    )
                } else {
                    context.default_metallic_roughness_image_id
                },
                emissive_texture_id: if let Some(emissive_texture) = material.emissive_texture() {
                    context.images.push(
                        load_image_from_gltf(
                            &emissive_texture.texture(),
                            true,
                            &buffers,
                            context,
                        )
                        .await,
                    )
                } else {
                    context.black_image_id
                },
                num_indices: indices.len() as u32,
            };

            primitives.push(primitive);
        }
    }

    Model { primitives }
}

async fn load_image_from_gltf(
    texture: &gltf::Texture<'_>,
    srgb: bool,
    buffers: &ModelBuffers<'_>,
    context: &ModelLoadContext<'_>,
) -> Resource<Texture> {
    let image = texture.source();
    let source = image.source();

    let bytes: std::borrow::Cow<[u8]> = match source {
        gltf::image::Source::View { view, mime_type } => {
            log::info!("{} {}", texture.index(), mime_type);
            let buffer = view.buffer();
            std::borrow::Cow::Borrowed(match buffer.source() {
                gltf::buffer::Source::Bin => buffers.blob.unwrap(),
                gltf::buffer::Source::Uri(_) => buffers.map.get(&buffer.index()).map(|vec| &vec[..]).unwrap(),
            })
        }
        gltf::image::Source::Uri { uri, mime_type } => {
            let url = url::Url::options()
                .base_url(Some(&context.url))
                .parse(uri)
                .unwrap();

            std::borrow::Cow::Owned(if url.scheme() == "data" {
                let (mime_type, data) = url.path().split_once(',').unwrap();
                log::info!("Got: {}", mime_type);
                base64::decode(data).unwrap()
            } else {
                fetch_bytes(&url).await
            })
        }
    };

    // todo: don't load the same image twice.

    let image = image::load_from_memory(&bytes).unwrap();

    let image = image.to_rgba8();

    let format = if srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };

    load_image(
        context.device,
        context.queue,
        image.width(),
        image.height(),
        format,
        &*image,
    )
}

fn load_image(
    device: &Device<(&'static str, u32, u32)>,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    bytes: &[u8],
) -> Resource<Texture> {
    let texture = device.inner.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
        },
        bytes,
    );

    device.create_resource(Texture {
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        texture,
    })
}

struct ModelPrimitive {
    indices: wgpu::Buffer,
    positions: wgpu::Buffer,
    normals: wgpu::Buffer,
    uvs: wgpu::Buffer,
    normal_texture_id: usize,
    albedo_texture_id: usize,
    metallic_roughness_texture_id: usize,
    emissive_texture_id: usize,
    num_indices: u32,
}

async fn fetch_bytes(url: &url::Url) -> Vec<u8> {
    let response: web_sys::Response = wasm_bindgen_futures::JsFuture::from(
        web_sys::window().unwrap().fetch_with_str(url.as_str()),
    )
    .await
    .unwrap()
    .into();

    log::info!(
        "{}",
        &response.headers().get("content-length").unwrap().unwrap()
    );

    let array_buffer: js_sys::ArrayBuffer =
        wasm_bindgen_futures::JsFuture::from(response.array_buffer().unwrap())
            .await
            .unwrap()
            .into();

    let uint8_buffer = js_sys::Uint8Array::new(&array_buffer);

    uint8_buffer.to_vec()
}

#[derive(Default)]
struct Images {
    inner: Vec<Resource<Texture>>,
}

impl Images {
    fn push(&mut self, image: Resource<Texture>) -> usize {
        let id = self.inner.len();
        self.inner.push(image);
        id
    }

    fn get(&self, id: usize) -> BindingResource {
        BindingResource::Texture(&self.inner[id])
    }
}
