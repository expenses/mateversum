use crate::components::{
    Instance, InstanceOf, InstanceRange, Instances, Model, ModelUrl, PendingModel,
};
use crate::{
    BindGroupLayouts, CompositeBindGroup, Device, IndexBuffer, InstanceBuffer,
    IntermediateColorFramebuffer, IntermediateDepthFramebuffer, LinearSampler, MainBindGroup,
    ModelUrls, Pipelines, Queue, SkyboxUniformBindGroup, SkyboxUniformBuffer, UniformBuffer,
    VertexBuffers,
};
use bevy_ecs::prelude::{Added, Commands, Entity, NonSend, Query, Res, ResMut};
use renderer_core::utils::Setter;
use renderer_core::{bytemuck, crevice::std140::AsStd140, shared_structs, Texture};
use std::sync::Arc;

pub(crate) mod rendering;

pub fn create_bind_group_layouts_and_pipelines(
    device: Res<Device>,
    pipeline_options: Res<renderer_core::PipelineOptions>,
    mut commands: Commands,
) {
    let device = &device.0;

    let bind_group_layouts = renderer_core::BindGroupLayouts::new(&device, &pipeline_options);

    let pipelines = renderer_core::Pipelines::new(&device, &bind_group_layouts, &pipeline_options);

    commands.insert_resource(BindGroupLayouts(Arc::new(bind_group_layouts)));
    commands.insert_resource(Pipelines(Arc::new(pipelines)));
    commands.insert_resource(IntermediateColorFramebuffer(None));
    commands.insert_resource(IntermediateDepthFramebuffer(None));
    commands.insert_resource(CompositeBindGroup(None));
}

pub fn clear_instance_buffer(
    mut instance_buffer: ResMut<InstanceBuffer>,
    mut query: Query<&mut Instances>,
) {
    instance_buffer.0.clear();

    query.for_each_mut(|mut instances| instances.0.clear());
}

// Here would be a good place to do culling.
pub fn push_entity_instances(
    mut instance_query: Query<(&InstanceOf, &Instance)>,
    mut model_query: Query<&mut Instances>,
) {
    instance_query.for_each_mut(|(instance_of, instance)| {
        match model_query.get_mut(instance_of.0) {
            Ok(mut instances) => {
                instances.0.push(instance.0);
            }
            Err(error) => {
                log::warn!("Got an error when pushing an instance: {}", error);
            }
        }
    })
}

pub fn upload_instances(
    device: Res<Device>,
    queue: Res<Queue>,
    mut instance_buffer: ResMut<InstanceBuffer>,
    mut query: Query<(&Instances, &mut InstanceRange)>,
) {
    let mut command_encoder = device
        .0
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("command encoder"),
        });

    query.for_each_mut(|(instances, mut instance_range)| {
        instance_range.0 =
            instance_buffer
                .0
                .push(&instances.0, &device.0, &queue.0, &mut command_encoder);
    });

    queue.0.submit(std::iter::once(command_encoder.finish()));
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

    let index_buffer = Arc::new(parking_lot::Mutex::new(renderer_core::IndexBuffer::new(
        1024, &device,
    )));

    let vertex_buffers = Arc::new(parking_lot::Mutex::new(renderer_core::VertexBuffers::new(
        1024, &device,
    )));

    let instance_buffer = renderer_core::InstanceBuffer::new(
        1,
        &device,
        wgpu::BufferUsages::VERTEX,
        "instance buffer",
    );

    commands.insert_resource(IndexBuffer(index_buffer.clone()));
    commands.insert_resource(VertexBuffers(vertex_buffers.clone()));
    commands.insert_resource(InstanceBuffer(instance_buffer));
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

pub fn start_loading_models(
    query: Query<(Entity, &ModelUrl), Added<ModelUrl>>,
    device: Res<Device>,
    queue: Res<Queue>,
    pipelines: Res<Pipelines>,
    bind_group_layouts: Res<BindGroupLayouts>,
    (index_buffer, vertex_buffers): (Res<IndexBuffer>, Res<VertexBuffers>),
    mut model_urls: ResMut<ModelUrls>,
    mut commands: Commands,
) {
    let device = &device.0;
    let queue = &queue.0;

    query.for_each(|(entity, url)| {
        let url = url.0.clone();
        let vertex_buffers = vertex_buffers.0.clone();
        let index_buffer = index_buffer.0.clone();

        let model_setter = Setter(Default::default());

        commands
            .entity(entity)
            .insert(PendingModel(model_setter.clone()));

        // Insert a link back from the url to the entity for lookups.
        model_urls.0.insert(url.clone(), entity);

        wasm_bindgen_futures::spawn_local({
            let device = device.clone();
            let queue = queue.clone();
            let bind_group_layouts = bind_group_layouts.0.clone();
            let pipelines = pipelines.0.clone();

            async move {
                let result = renderer_core::assets::models::Model::load(
                    &renderer_core::assets::models::Context {
                        device,
                        queue,
                        bind_group_layouts,
                        http_client: super::SimpleHttpClient,
                        index_buffer,
                        vertex_buffers,
                        pipelines,
                    },
                    &url,
                )
                .await;

                match result {
                    Err(error) => {
                        log::error!(
                            "Got an error while trying to load a model from '{}': {}",
                            url,
                            error
                        );
                    }
                    Ok(model) => {
                        model_setter.set(model);
                    }
                }
            }
        })
    })
}

pub fn finish_loading_models(query: Query<(Entity, &PendingModel)>, mut commands: Commands) {
    query.for_each(|(entity, pending_model)| {
        if let Some(mut lock) = pending_model.0 .0.try_lock() {
            if let Some(loaded_model) = lock.take() {
                commands.entity(entity).insert(Model(loaded_model));
            }
        }
    })
}
