use crate::*;

use crate::components::{InstanceRange, Model};
use bevy_ecs::prelude::{Local, NonSend, Query, Res, ResMut};
use renderer_core::assets::models::PrimitiveRanges;
use renderer_core::create_view_from_device_framebuffer;
use renderer_core::utils::BorrowedOrOwned;

pub(crate) fn render(
    frame: NonSend<web_sys::XrFrame>,
    device: Res<Device>,
    queue: Res<Queue>,
    pipelines: Res<Pipelines>,
    bind_group_layouts: Res<BindGroupLayouts>,
    mut main_bind_group: ResMut<MainBindGroup>,
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
    mut models: Query<(&mut Model, &InstanceRange)>,
    mut model_bind_groups: Local<ModelBindGroups>,
) {
    let device = &device.0;
    let queue = &queue.0;
    let pipelines = &pipelines.0;
    let bind_group_layouts = &bind_group_layouts.0;
    let main_bind_group = main_bind_group.0.get();

    // These `.lock`s looks scary, but it will never actually block (in wasm)
    // because that would panic the main thread otherwise!
    let vertex_buffers = &vertex_buffers.0.lock();
    let index_buffer = &index_buffer.0.lock();

    let xr_session: web_sys::XrSession = frame.session();

    let base_layer = xr_session.render_state().base_layer().unwrap();

    let framebuffer: web_sys::WebGlFramebuffer =
        js_sys::Reflect::get(&base_layer, &"framebuffer".into())
            .unwrap()
            .into();

    let framebuffer_colour_attachment = create_view_from_device_framebuffer(
        device,
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
            device,
            framebuffer,
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

    model_bind_groups.collect(&mut models);

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

        render_pass.set_bind_group(0, main_bind_group, &[]);

        {
            render_pass.set_pipeline(&pipelines.pbr.opaque);

            render_all_primitives(
                &mut render_pass,
                &models,
                &model_bind_groups,
                |primitive_ranges| primitive_ranges.opaque.clone(),
            );

            render_pass.set_pipeline(&pipelines.pbr_double_sided.opaque);

            render_all_primitives(
                &mut render_pass,
                &models,
                &model_bind_groups,
                |primitive_ranges| primitive_ranges.opaque_double_sided.clone(),
            );

            render_pass.set_pipeline(&pipelines.pbr.alpha_clipped);

            render_all_primitives(
                &mut render_pass,
                &models,
                &model_bind_groups,
                |primitive_ranges| primitive_ranges.alpha_clipped.clone(),
            );

            render_pass.set_pipeline(&pipelines.pbr_double_sided.alpha_clipped);

            render_all_primitives(
                &mut render_pass,
                &models,
                &model_bind_groups,
                |primitive_ranges| primitive_ranges.alpha_clipped_double_sided.clone(),
            );
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

        render_pass.set_bind_group(0, main_bind_group, &[]);
        render_pass.set_bind_group(1, composite_bind_group, &[]);

        render_pass.draw(0..3, 0..1);

        drop(render_pass);
    }

    queue.submit(std::iter::once(command_encoder.finish()));
}

// The model bind groups for the current frame
#[derive(Default)]
pub struct ModelBindGroups {
    bind_groups: Vec<Arc<wgpu::BindGroup>>,
    // We use a `Vec` of offsets here to avoid needing a `Vec<Vec<Arc<wgpu::BindGroup>>>`
    // This means we can just clear the `Vec`s instead of re-allocating.
    offsets: Vec<usize>,
}

impl ModelBindGroups {
    fn collect(&mut self, query: &mut Query<(&mut Model, &InstanceRange)>) {
        self.bind_groups.clear();
        self.offsets.clear();

        // This is mutable because it involves potentially swapping out the dummy bind groups
        // for loaded ones.
        query.for_each_mut(|(mut model, _)| {
            self.offsets.push(self.bind_groups.len());

            // Todo: we could do a check if the model has any instances here
            // and not write the bind groups if not, which would mean that we don't have to do a check
            // however many times we do a `render_all_primitives` call. But that'd be less clear and
            // I'm not sure if it's worthwhile.
            self.bind_groups.extend(
                model
                    .0
                    .primitives
                    .iter_mut()
                    .map(|primitive| primitive.bind_group.get().clone()),
            );
        })
    }

    fn bind_groups_for_model(&self, model_index: usize) -> &[Arc<wgpu::BindGroup>] {
        &self.bind_groups[self.offsets[model_index]..]
    }
}

fn render_all_primitives<'a, G: Fn(&PrimitiveRanges) -> Range<usize>>(
    render_pass: &mut wgpu::RenderPass<'a>,
    models: &Query<(&mut Model, &InstanceRange)>,
    model_bind_groups: &'a ModelBindGroups,
    primitive_range_getter: G,
) {
    for (model_index, (model, instance_range)) in models.iter().enumerate() {
        // Don't issue commands for models with no (visible) instances.
        if !instance_range.0.is_empty() {
            // Get the range of primtives we're rendering
            let range = primitive_range_getter(&model.0.primitive_ranges);

            // Get the primitives we're rendering
            let primitives = &model.0.primitives[range.clone()];
            // And their associated material bind groups
            let bind_groups = &model_bind_groups.bind_groups_for_model(model_index)[range];

            for (primitive, bind_group) in primitives.iter().zip(bind_groups) {
                render_pass.set_bind_group(1, bind_group, &[]);

                render_pass.draw_indexed(
                    primitive.index_buffer_range.clone(),
                    0,
                    instance_range.0.clone(),
                );
            }
        }
    }
}
