use crate::{
    BindGroupLayouts, CompositeBindGroup, Device, IndexBuffer, InstanceBuffer,
    IntermediateColorFramebuffer, IntermediateDepthFramebuffer, LinearSampler, MainBindGroup,
    Model, Pipelines, Queue, SkyboxUniformBindGroup, SkyboxUniformBuffer, TestModel,
    TestModelBindGroup, UniformBuffer, VertexBuffers,
};
use bevy_ecs::prelude::{Commands, NonSend, Res, ResMut};
use renderer_core::glam::{Vec3, Vec4};
use renderer_core::{
    bytemuck, create_view_from_device_framebuffer, crevice::std140::AsStd140, shared_structs,
    Instance, Texture,
};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub fn create_bind_group_layouts_and_pipelines(
    device: Res<Device>,
    pipeline_options: Res<renderer_core::PipelineOptions>,
    mut commands: Commands,
) {
    let device = &device.0;

    let bind_group_layouts = renderer_core::BindGroupLayouts::new(&device, &pipeline_options);

    // todo: this probably only gives very minimal gains.
    let shader_cache: renderer_core::ResourceCache<wgpu::ShaderModule> = Default::default();

    let pipelines = renderer_core::Pipelines::new(
        &device,
        &shader_cache,
        &bind_group_layouts,
        &pipeline_options,
    );

    commands.insert_resource(BindGroupLayouts(Arc::new(bind_group_layouts)));
    commands.insert_resource(Pipelines(Arc::new(pipelines)));
    commands.insert_resource(IntermediateColorFramebuffer(None));
    commands.insert_resource(IntermediateDepthFramebuffer(None));
    commands.insert_resource(CompositeBindGroup(None));
}

pub fn allocate_bind_groups(
    device: Res<Device>,
    queue: Res<Queue>,
    pipelines: Res<Pipelines>,
    bind_group_layouts: Res<BindGroupLayouts>,
    mut commands: Commands,
) {
    let device = &device.0;
    let queue = &queue.0;
    let pipelines = &pipelines.0;
    let bind_group_layouts = &bind_group_layouts.0;

    let ibl_lut_texture = Arc::new(Texture::new(device.create_texture(
        &wgpu::TextureDescriptor {
            label: Some("dummy IBL LUT"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            sample_count: 1,
            mip_level_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            format: wgpu::TextureFormat::Rgba16Float,
        },
    )));

    #[rustfmt::skip]
    let ibl_diffuse_cubemap_texture = Arc::new(Texture::new_cubemap(device.create_texture(
        &wgpu::TextureDescriptor {
            label: Some("dummy diffuse cubemap"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            sample_count: 1,
            mip_level_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            format: wgpu::TextureFormat::Rgba16Float,
        }
    )));

    #[rustfmt::skip]
    let ibl_specular_cubemap_texture = Arc::new(Texture::new_cubemap(device.create_texture(
        &wgpu::TextureDescriptor {
            label: Some("dummy specular cubemap"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            sample_count: 1,
            mip_level_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            format: wgpu::TextureFormat::Rgba16Float,
        }
    )));

    let uniform_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform buffer"),
        size: std::mem::size_of::<<shared_structs::Uniforms as AsStd140>::Output>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    }));

    let linear_sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        anisotropy_clamp: Some(std::num::NonZeroU8::new(16).unwrap()), //performance_settings.anisotropy_clamp(),
        ..Default::default()
    }));

    let skybox_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("skybox uniform buffer"),
        size: std::mem::size_of::<<shared_structs::SkyboxUniforms as AsStd140>::Output>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let skybox_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("skybox uniform bind group"),
        layout: &bind_group_layouts.skybox,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: skybox_uniform_buffer.as_entire_binding(),
        }],
    });

    let main_bind_group = Arc::new(parking_lot::Mutex::new(device.create_bind_group(
        &wgpu::BindGroupDescriptor {
            label: Some("main bind group"),
            layout: &bind_group_layouts.uniform,
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
                    resource: wgpu::BindingResource::TextureView(&ibl_lut_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&ibl_diffuse_cubemap_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        &ibl_specular_cubemap_texture.view,
                    ),
                },
            ],
        },
    )));

    commands.insert_resource(UniformBuffer(uniform_buffer.clone()));
    commands.insert_resource(MainBindGroup(main_bind_group.clone()));
    commands.insert_resource(LinearSampler(linear_sampler.clone()));

    commands.insert_resource(SkyboxUniformBuffer(skybox_uniform_buffer));
    commands.insert_resource(SkyboxUniformBindGroup(skybox_uniform_bind_group));

    wasm_bindgen_futures::spawn_local({
        let device = device.clone();
        let queue = queue.clone();
        let pipelines = pipelines.clone();
        let bind_group_layouts = bind_group_layouts.clone();
        let linear_sampler = linear_sampler.clone();

        async move {
            let specular_url = url::Url::parse(
                "https://expenses.github.io/mateversum-web/environment_maps/helipad/specular_compressed.ktx2",
            )
            .unwrap();

            let specular_fut = renderer_core::assets::textures::load_ktx2_cubemap(
                renderer_core::assets::textures::Context {
                    device: device.clone(),
                    queue: queue.clone(),
                    http_client: super::SimpleHttpClient,
                    bind_group_layouts: bind_group_layouts.clone(),
                    pipelines: pipelines.clone(),
                },
                &specular_url,
            );

            let diffuse_url = url::Url::parse(
                "https://expenses.github.io/mateversum-web/environment_maps/helipad/diffuse_compressed.ktx2",
            )
            .unwrap();

            let diffuse_fut = renderer_core::assets::textures::load_ktx2_cubemap(
                renderer_core::assets::textures::Context {
                    device: device.clone(),
                    queue,
                    http_client: super::SimpleHttpClient,
                    bind_group_layouts: bind_group_layouts.clone(),
                    pipelines,
                },
                &diffuse_url,
            );

            let results = futures::future::join(specular_fut, diffuse_fut).await;

            match results {
                (Ok(ibl_specular_cubemap_texture), Ok(ibl_diffuse_cubemap_texture)) => {
                    // Bind groups are immutable so we need to rebuild it.
                    *main_bind_group.lock() =
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("main bind group"),
                            layout: &bind_group_layouts.uniform,
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
                                    resource: wgpu::BindingResource::TextureView(
                                        &ibl_lut_texture.view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::TextureView(
                                        &ibl_diffuse_cubemap_texture.view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::TextureView(
                                        &ibl_specular_cubemap_texture.view,
                                    ),
                                },
                            ],
                        });
                }
                _ => {
                    log::error!("Got an error while trying to load the cubemaps");
                }
            }
        }
    });

    fn load_single_pixel_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        bytes: &[u8; 4],
    ) -> Texture {
        Texture::new(device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
            },
            bytes,
        ))
    }

    let black_image = load_single_pixel_image(
        &*device,
        &queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &[0, 0, 0, 255],
    );
    let default_metallic_roughness_image = load_single_pixel_image(
        &*device,
        &queue,
        wgpu::TextureFormat::Rgba8Unorm,
        &[0, 255, 0, 255],
    );
    let flat_normals_image = load_single_pixel_image(
        &*device,
        &queue,
        wgpu::TextureFormat::Rgba8Unorm,
        &[127, 127, 255, 255],
    );
    let white_image = load_single_pixel_image(
        &*device,
        &queue,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &[255, 255, 255, 255],
    );

    let material_settings = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("material settings"),
        contents: bytemuck::bytes_of(
            &shared_structs::MaterialSettings {
                base_color_factor: Vec4::ONE,
                emissive_factor: Vec3::ZERO,
                metallic_factor: 0.0,
                roughness_factor: 1.0,
                is_unlit: false as u32,
            }
            .as_std140(),
        ),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    commands.insert_resource(TestModelBindGroup(device.create_bind_group(
        &wgpu::BindGroupDescriptor {
            label: Some("test model bind group"),
            layout: &bind_group_layouts.model,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&white_image.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&flat_normals_image.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &default_metallic_roughness_image.view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&black_image.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: material_settings.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
            ],
        },
    )));

    let index_buffer = Arc::new(parking_lot::Mutex::new(renderer_core::IndexBuffer::new(
        1024, &device,
    )));

    let vertex_buffers = Arc::new(parking_lot::Mutex::new(renderer_core::VertexBuffers::new(
        1024, &device,
    )));

    let instance_buffer = {
        let mut instance_buffer = renderer_core::InstanceBuffer::new(
            1024,
            &device,
            wgpu::BufferUsages::VERTEX,
            "instance buffer",
        );

        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("command encoder"),
        });

        instance_buffer.push(
            &[Instance::new(
                Vec3::new(0.0, 1.0, -2.0),
                1.0,
                Default::default(),
            )],
            &device,
            &queue,
            &mut command_encoder,
        );

        queue.submit(std::iter::once(command_encoder.finish()));

        instance_buffer
    };

    let test_model = Arc::new(parking_lot::Mutex::new(Model::default()));

    commands.insert_resource(TestModel(test_model.clone()));
    commands.insert_resource(IndexBuffer(index_buffer.clone()));
    commands.insert_resource(VertexBuffers(vertex_buffers.clone()));
    commands.insert_resource(InstanceBuffer(instance_buffer));

    wasm_bindgen_futures::spawn_local({
        let url = url::Url::parse(
            "https://expenses.github.io/mateversum-web/glTF-Sample-Models/2.0/DamagedHelmet/glTF/DamagedHelmet.gltf",
        )
        .unwrap();

        let device = device.clone();
        let queue = queue.clone();

        async move {
            let result = renderer_core::assets::models::Model::load(
                &renderer_core::assets::models::Context {
                    device,
                    queue,
                    http_client: super::SimpleHttpClient,
                    index_buffer,
                    vertex_buffers,
                },
                &url,
            )
            .await;

            match result {
                Err(error) => {
                    log::error!(
                        "Got an error while trying to load the test model: {}",
                        error
                    );
                }
                Ok(model) => {
                    *test_model.lock() = model;
                }
            }
        }
    })
}

pub fn update_uniform_buffers(
    pose: NonSend<web_sys::XrViewerPose>,
    pipeline_options: Res<renderer_core::PipelineOptions>,
    queue: Res<Queue>,
    uniform_buffer: Res<UniformBuffer>,
    skybox_uniform_buffer: Res<SkyboxUniformBuffer>,
) {
    let queue = &queue.0;

    use renderer_core::glam::Mat4;

    let parse_matrix = |vec| Mat4::from_cols_array(&<[f32; 16]>::try_from(vec).unwrap());

    let views = pose.views();

    let mut views_iter = views.iter();

    let left_view: web_sys::XrView = views_iter.next().unwrap().into();

    let left_proj = parse_matrix(left_view.projection_matrix());
    let left_inv = parse_matrix(left_view.transform().matrix()).inverse();

    let left_projection_view: renderer_core::shared_structs::FlatMat4 =
        (left_proj * left_inv).into();

    let left_instance = renderer_core::Instance::from_transform(left_view.transform(), 0.0);

    let (right_projection_view, right_proj, right_instance) = if let Some(right_view) =
        views_iter.next()
    {
        let right_view: web_sys::XrView = right_view.into();

        let right_proj = parse_matrix(right_view.projection_matrix());
        let right_inv = parse_matrix(right_view.transform().matrix()).inverse();

        let right_projection_view: renderer_core::shared_structs::FlatMat4 =
            (right_proj * right_inv).into();

        let right_instance = renderer_core::Instance::from_transform(right_view.transform(), 0.0);

        (right_projection_view, right_proj, right_instance)
    } else {
        Default::default()
    };

    let uniforms = renderer_core::shared_structs::Uniforms {
        left_projection_view,
        right_projection_view,
        left_eye_position: left_instance.position,
        right_eye_position: right_instance.position,
        render_direct_to_framebuffer: pipeline_options.render_direct_to_framebuffer() as u32,
        inline_tonemapping: pipeline_options.inline_tonemapping as u32,
    };

    queue.write_buffer(
        &uniform_buffer.0,
        0,
        renderer_core::bytemuck::bytes_of(&uniforms.as_std140()),
    );

    let skybox_uniforms = shared_structs::SkyboxUniforms {
        left_projection_inverse: left_proj.inverse().into(),
        right_projection_inverse: right_proj.inverse().into(),
        left_view_inverse: left_instance.rotation.into(),
        right_view_inverse: right_instance.rotation.into(),
    };

    queue.write_buffer(
        &skybox_uniform_buffer.0,
        0,
        bytemuck::bytes_of(&skybox_uniforms.as_std140()),
    );
}

pub fn render(
    frame: NonSend<web_sys::XrFrame>,
    device: Res<Device>,
    queue: Res<Queue>,
    pipelines: Res<Pipelines>,
    bind_group_layouts: Res<BindGroupLayouts>,
    main_bind_group: Res<MainBindGroup>,
    skybox_uniform_bind_group: Res<SkyboxUniformBindGroup>,
    mut intermediate_color_framebuffer: ResMut<IntermediateColorFramebuffer>,
    mut intermediate_depth_framebuffer: ResMut<IntermediateDepthFramebuffer>,
    mut composite_bind_group: ResMut<CompositeBindGroup>,
    pipeline_options: Res<renderer_core::PipelineOptions>,
    linear_sampler: Res<LinearSampler>,
    (index_buffer, vertex_buffers, instance_buffer): (
        Res<IndexBuffer>,
        Res<VertexBuffers>,
        Res<InstanceBuffer>,
    ),
    test_model: Res<TestModel>,
    test_model_bind_group: Res<TestModelBindGroup>,
) {
    let device = &device.0;
    let queue = &queue.0;
    let pipelines = &pipelines.0;
    let bind_group_layouts = &bind_group_layouts.0;

    // These `.lock`s looks scary, but it will never actually block (in wasm)
    // because that would panic the main thread otherwise!
    let main_bind_group = &main_bind_group.0.lock();
    let vertex_buffers = &vertex_buffers.0.lock();
    let index_buffer = &index_buffer.0.lock();
    let test_model = &test_model.0.lock();

    let xr_session: web_sys::XrSession = frame.session();

    let base_layer = xr_session.render_state().base_layer().unwrap();

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

    let num_views = pipeline_options
        .multiview
        .map(|views| views.get())
        .unwrap_or(1);

    let (intermediate_color_framebuffer, composite_bind_group) =
        if pipeline_options.render_direct_to_framebuffer() {
            (None, None)
        } else {
            let intermediate_color_framebuffer =
                intermediate_color_framebuffer.0.get_or_insert_with(|| {
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("intermediate color framebuffer"),
                        size: wgpu::Extent3d {
                            width: base_layer.framebuffer_width() / num_views,
                            height: base_layer.framebuffer_height(),
                            depth_or_array_layers: num_views,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        // Always true at the moment.
                        format: if pipeline_options.inline_tonemapping {
                            wgpu::TextureFormat::Rgba8Unorm
                        } else {
                            wgpu::TextureFormat::Rgba16Float
                        },
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    });

                    let view = texture.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(if pipeline_options.multiview.is_none() {
                            wgpu::TextureViewDimension::D2
                        } else {
                            wgpu::TextureViewDimension::D2Array
                        }),
                        ..Default::default()
                    });

                    renderer_core::Texture { texture, view }
                });

            let composite_bind_group = composite_bind_group.0.get_or_insert_with(|| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("composite bind group"),
                    layout: &bind_group_layouts.tonemap,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Sampler(&linear_sampler.0),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &intermediate_color_framebuffer.view,
                            ),
                        },
                    ],
                })
            });

            (
                Some(intermediate_color_framebuffer),
                Some(composite_bind_group),
            )
        };

    let depth_attachment = if pipeline_options.render_direct_to_framebuffer() {
        BorrowedOrOwned::Owned(create_view_from_device_framebuffer(
            &device,
            framebuffer.clone(),
            &base_layer,
            wgpu::TextureFormat::Depth24PlusStencil8,
            "device framebuffer (depth)",
        ))
    } else {
        BorrowedOrOwned::Borrowed(intermediate_depth_framebuffer.0.get_or_insert_with(|| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("intermediate depth framebuffer"),
                size: wgpu::Extent3d {
                    width: base_layer.framebuffer_width() / num_views,
                    height: base_layer.framebuffer_height(),
                    depth_or_array_layers: num_views,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
            });

            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(if pipeline_options.multiview.is_none() {
                    wgpu::TextureViewDimension::D2
                } else {
                    wgpu::TextureViewDimension::D2Array
                }),
                ..Default::default()
            });

            renderer_core::Texture { texture, view }
        }))
    };

    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("command encoder"),
    });

    let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("main render pass"),
        color_attachments: &[wgpu::RenderPassColorAttachment {
            view: if let Some(intermediate_color_framebuffer) = intermediate_color_framebuffer {
                &intermediate_color_framebuffer.view
            } else {
                &framebuffer_colour_attachment.view
            },
            resolve_target: None,
            ops: wgpu::Operations {
                // Note: when rendering to a Quest 2, clearing the intermediate framebuffer
                // makes the skybox only render on one eye! No clue why.
                load: wgpu::LoadOp::Load,
                store: true,
            },
        }],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &depth_attachment.borrow().view,
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

    {
        render_pass.set_index_buffer(index_buffer.buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.set_vertex_buffer(0, vertex_buffers.position.slice(..));
        render_pass.set_vertex_buffer(1, vertex_buffers.normal.slice(..));
        render_pass.set_vertex_buffer(2, vertex_buffers.uv.slice(..));
        render_pass.set_vertex_buffer(3, instance_buffer.0.buffer.slice(..));

        render_pass.set_bind_group(0, &main_bind_group, &[]);

        render_pass.set_pipeline(&pipelines.pbr.opaque);
        render_pass.set_bind_group(1, &test_model_bind_group.0, &[]);
        for primitive_index in test_model.primitive_ranges.opaque.clone() {
            let primitive = &test_model.primitives[primitive_index];

            render_pass.draw_indexed(primitive.index_buffer_range.clone(), 0, 0..1);
        }

        render_pass.set_pipeline(&pipelines.skybox);
        render_pass.set_bind_group(1, &skybox_uniform_bind_group.0, &[]);
        render_pass.draw(0..3, 0..1);
    }

    drop(render_pass);

    if let Some(composite_bind_group) = composite_bind_group {
        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite render pass"),
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

        render_pass.set_bind_group(0, &main_bind_group, &[]);
        render_pass.set_bind_group(1, composite_bind_group, &[]);

        render_pass.draw(0..3, 0..1);

        drop(render_pass);
    }

    queue.submit(std::iter::once(command_encoder.finish()));
}

// std::borrow::Cow has too many type restrictions to use instead of this.
// There's probably something in the std library that does the same thing tho?
enum BorrowedOrOwned<'a, T> {
    Owned(T),
    Borrowed(&'a T),
}

impl<'a, T> BorrowedOrOwned<'a, T> {
    fn borrow(&'a self) -> &'a T {
        match self {
            Self::Owned(value) => &value,
            Self::Borrowed(reference) => reference,
        }
    }
}
