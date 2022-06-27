use crate::bind_group_layouts::BindGroupLayouts;

pub struct PipelineOptions {
    pub multiview: Option<std::num::NonZeroU32>,
    pub inline_tonemapping: bool,
}

impl PipelineOptions {
    // If we're not doing multiview rendering or a seperate tonemapping pass then we can render
    // meshes directly to the device framebuffer.
    pub fn render_direct_to_framebuffer(&self) -> bool {
        self.multiview.is_none() && self.inline_tonemapping
    }
}

pub struct Pipelines {
    pub pbr: PipelineSet,
    pub pbr_double_sided: PipelineSet,
    pub stencil_write: wgpu::RenderPipeline,
    pub set_depth: wgpu::RenderPipeline,
    pub tonemap: wgpu::RenderPipeline,
    pub ui: wgpu::RenderPipeline,
    pub skybox: wgpu::RenderPipeline,
    pub skybox_mirrored: wgpu::RenderPipeline,
    pub bc6h_decompression: wgpu::RenderPipeline,
    pub blit: wgpu::RenderPipeline,
    pub srgb_blit: wgpu::RenderPipeline,
}

impl Pipelines {
    pub fn new(
        device: &wgpu::Device,
        bind_group_layouts: &BindGroupLayouts,
        options: &PipelineOptions,
    ) -> Self {
        let target_format = if options.inline_tonemapping {
            wgpu::TextureFormat::Rgba8Unorm
        } else {
            wgpu::TextureFormat::Rgba16Float
        };

        let flip_viewport = options.render_direct_to_framebuffer();

        let front_face = if flip_viewport {
            wgpu::FrontFace::Cw
        } else {
            wgpu::FrontFace::Ccw
        };

        let multiview = options.multiview;

        let uniform_only_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("uniform only pipeline layout"),
                bind_group_layouts: &[&bind_group_layouts.uniform],
                push_constant_ranges: &[],
            });

        let model_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("model pipeline layout"),
                bind_group_layouts: &[&bind_group_layouts.uniform, &bind_group_layouts.model],
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

        let prefix = if multiview.is_none() {
            "single_view::"
        } else {
            ""
        };

        let vertex_state = wgpu::VertexState {
            module: &device.create_shader_module(&if multiview.is_none() {
                wgpu::include_spirv!("../../compiled-shaders/single_view_vertex.spv")
            } else {
                wgpu::include_spirv!("../../compiled-shaders/vertex.spv")
            }),
            entry_point: &format!("{}vertex", prefix),
            buffers: vertex_buffers,
        };

        let normal_primitive_state = wgpu::PrimitiveState {
            front_face,
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

        let fullscreen_tri_vertex_state = wgpu::VertexState {
            module: &device.create_shader_module(&wgpu::include_spirv!(
                "../../compiled-shaders/fullscreen_tri.spv"
            )),
            entry_point: "fullscreen_tri",
            buffers: &[],
        };

        let tonemap_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layouts.uniform, &bind_group_layouts.tonemap],
                push_constant_ranges: &[],
            });

        let tonemap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tonemap pipeline"),
            layout: Some(&tonemap_pipeline_layout),
            vertex: fullscreen_tri_vertex_state.clone(),
            fragment: Some(wgpu::FragmentState {
                module: &device.create_shader_module(&if multiview.is_none() {
                    wgpu::include_spirv!("../../compiled-shaders/single_view_tonemap.spv")
                } else {
                    wgpu::include_spirv!("../../compiled-shaders/tonemap.spv")
                }),
                entry_point: &format!("{}tonemap", prefix),
                targets: &[wgpu::TextureFormat::Rgba8Unorm.into()],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: Default::default(),
        });

        let flat_blue_test_fragment_shader = device.create_shader_module(&wgpu::include_spirv!(
            "../../compiled-shaders/flat_blue.spv"
        ));

        let stencil_write = wgpu::StencilFaceState {
            compare: wgpu::CompareFunction::Always,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::IncrementClamp,
        };

        let stencil_write_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("stencil write pipeline"),
                layout: Some(&uniform_only_pipeline_layout),
                vertex: vertex_state.clone(),

                fragment: Some(wgpu::FragmentState {
                    module: &flat_blue_test_fragment_shader,
                    entry_point: "flat_blue",
                    targets: &[wgpu::ColorTargetState {
                        format: target_format,
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
            layout: Some(&uniform_only_pipeline_layout),
            vertex: vertex_state.clone(),

            fragment: Some(wgpu::FragmentState {
                module: &flat_blue_test_fragment_shader,
                entry_point: "flat_blue",
                targets: &[wgpu::ColorTargetState {
                    format: target_format,
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
                bind_group_layouts: &[
                    &bind_group_layouts.uniform,
                    &bind_group_layouts.model,
                    &bind_group_layouts.mirror_uniform,
                ],
                push_constant_ranges: &[],
            });

        let mirrored_vertex = wgpu::VertexState {
            module: &device.create_shader_module(&if multiview.is_none() {
                wgpu::include_spirv!("../../compiled-shaders/single_view_vertex_mirrored.spv")
            } else {
                wgpu::include_spirv!("../../compiled-shaders/vertex_mirrored.spv")
            }),
            entry_point: &format!("{}vertex_mirrored", prefix),
            buffers: vertex_buffers,
        };

        let normal_primitive_state = wgpu::PrimitiveState {
            front_face,
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

        let ui_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui pipeline layout"),
            bind_group_layouts: &[&bind_group_layouts.uniform, &bind_group_layouts.ui_texture],
            push_constant_ranges: &[],
        });

        let skybox_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("skybox pipeline layout"),
                bind_group_layouts: &[&bind_group_layouts.uniform, &bind_group_layouts.skybox],
                push_constant_ranges: &[],
            });

        let skybox_mirrored_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("skybox mirrored pipeline layout"),
                bind_group_layouts: &[
                    &bind_group_layouts.uniform,
                    &bind_group_layouts.skybox,
                    &bind_group_layouts.mirror_uniform,
                ],
                push_constant_ranges: &[],
            });

        let fragment_opaque = wgpu::FragmentState {
            module: &device.create_shader_module(&if multiview.is_none() {
                wgpu::include_spirv!("../../compiled-shaders/single_view_fragment.spv")
            } else {
                wgpu::include_spirv!("../../compiled-shaders/fragment.spv")
            }),
            entry_point: &format!("{}fragment", prefix),
            targets: &[target_format.into()],
        };

        let fragment_alpha_clipped = wgpu::FragmentState {
            module: &device.create_shader_module(&if multiview.is_none() {
                wgpu::include_spirv!(
                    "../../compiled-shaders/single_view_fragment_alpha_clipped.spv"
                )
            } else {
                wgpu::include_spirv!("../../compiled-shaders/fragment_alpha_clipped.spv")
            }),
            entry_point: &format!("{}fragment_alpha_clipped", prefix),
            targets: &[target_format.into()],
        };

        let bc6h_decompression_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bc6h decompression pipeline layout"),
                bind_group_layouts: &[&bind_group_layouts.uint_texture],
                push_constant_ranges: &[],
            });

        let bc6h_decompression_target = wgpu::TextureFormat::Rg11b10Float;

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layouts.sampled_texture],
            push_constant_ranges: &[],
        });

        let blit_fragment_shader =
            device.create_shader_module(&wgpu::include_spirv!("../../compiled-shaders/blit.spv"));

        let skybox_fragment_shader = device.create_shader_module(&wgpu::include_spirv!(
            "../../compiled-shaders/fragment_skybox.spv"
        ));

        Self {
            pbr: PipelineSet::new(
                device,
                &model_pipeline_layout,
                &mirrored_pipeline_layout,
                vertex_state.clone(),
                mirrored_vertex.clone(),
                fragment_opaque.clone(),
                fragment_alpha_clipped.clone(),
                multiview,
                false,
                front_face,
            ),
            pbr_double_sided: PipelineSet::new(
                device,
                &model_pipeline_layout,
                &mirrored_pipeline_layout,
                vertex_state.clone(),
                mirrored_vertex.clone(),
                fragment_opaque.clone(),
                fragment_alpha_clipped.clone(),
                multiview,
                true,
                front_face,
            ),
            stencil_write: stencil_write_pipeline,
            set_depth: set_depth_pipeline,
            tonemap: tonemap_pipeline,
            ui: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ui pipeline"),
                layout: Some(&ui_pipeline_layout),
                vertex: vertex_state.clone(),
                fragment: Some(wgpu::FragmentState {
                    module: &device.create_shader_module(&wgpu::include_spirv!(
                        "../../compiled-shaders/fragment_ui.spv"
                    )),
                    entry_point: "fragment_ui",
                    targets: &[wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }],
                }),
                primitive: normal_primitive_state,
                depth_stencil: Some(normal_depth_state),
                multisample: Default::default(),
                multiview,
            }),
            skybox: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("skybox pipeline"),
                layout: Some(&skybox_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &device.create_shader_module(&if multiview.is_none() {
                        wgpu::include_spirv!("../../compiled-shaders/single_view_vertex_skybox.spv")
                    } else {
                        wgpu::include_spirv!("../../compiled-shaders/vertex_skybox.spv")
                    }),
                    entry_point: &format!("{}vertex_skybox", prefix),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &skybox_fragment_shader,
                    entry_point: "fragment_skybox",
                    targets: &[target_format.into()],
                }),
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    bias: wgpu::DepthBiasState::default(),
                    stencil: wgpu::StencilState::default(),
                }),
                multisample: Default::default(),
                multiview,
            }),
            skybox_mirrored: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("skybox mirrored pipeline"),
                layout: Some(&skybox_mirrored_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &device.create_shader_module(&if multiview.is_none() {
                        wgpu::include_spirv!(
                            "../../compiled-shaders/single_view_vertex_skybox_mirrored.spv"
                        )
                    } else {
                        wgpu::include_spirv!("../../compiled-shaders/vertex_skybox_mirrored.spv")
                    }),
                    entry_point: &format!("{}vertex_skybox_mirrored", prefix),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &skybox_fragment_shader,
                    entry_point: "fragment_skybox",
                    targets: &[target_format.into()],
                }),
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    bias: wgpu::DepthBiasState::default(),
                    stencil: wgpu::StencilState::default(),
                }),
                multisample: Default::default(),
                multiview,
            }),
            bc6h_decompression: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&bc6h_decompression_pipeline_layout),
                vertex: fullscreen_tri_vertex_state.clone(),
                fragment: Some(wgpu::FragmentState {
                    module: &device.create_shader_module(&wgpu::include_spirv!(
                        "../../compiled-shaders/bc6.spv"
                    )),
                    entry_point: "main",
                    targets: &[bc6h_decompression_target.into()],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: Default::default(),
            }),
            blit: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("blit pipeline"),
                layout: Some(&blit_pipeline_layout),
                vertex: fullscreen_tri_vertex_state.clone(),
                fragment: Some(wgpu::FragmentState {
                    module: &blit_fragment_shader,
                    entry_point: "blit",
                    targets: &[wgpu::TextureFormat::Rgba8Unorm.into()],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: Default::default(),
            }),
            srgb_blit: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("srgb blit pipeline"),
                layout: Some(&blit_pipeline_layout),
                vertex: fullscreen_tri_vertex_state.clone(),
                fragment: Some(wgpu::FragmentState {
                    module: &blit_fragment_shader,
                    entry_point: "blit",
                    targets: &[wgpu::TextureFormat::Rgba8UnormSrgb.into()],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: Default::default(),
            }),
        }
    }
}

pub struct PipelineSet {
    pub opaque: wgpu::RenderPipeline,
    pub alpha_clipped: wgpu::RenderPipeline,
    pub opaque_mirrored: wgpu::RenderPipeline,
    pub alpha_clipped_mirrored: wgpu::RenderPipeline,
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
        double_sided: bool,
        front_face: wgpu::FrontFace,
    ) -> Self {
        let normal_primitive_state = wgpu::PrimitiveState {
            front_face,
            cull_mode: if !double_sided {
                Some(wgpu::Face::Back)
            } else {
                None
            },
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
            front_face: match front_face {
                wgpu::FrontFace::Ccw => wgpu::FrontFace::Cw,
                wgpu::FrontFace::Cw => wgpu::FrontFace::Ccw,
            },
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        };

        Self {
            opaque: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("opaque pipeline"),
                layout: Some(pipeline_layout),
                vertex: normal_vertex.clone(),
                fragment: Some(opaque_fragment.clone()),
                primitive: normal_primitive_state,
                depth_stencil: Some(normal_depth_state.clone()),
                multisample: Default::default(),
                multiview,
            }),
            alpha_clipped: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("alpha clipped pipeline"),
                layout: Some(pipeline_layout),
                vertex: normal_vertex.clone(),
                fragment: Some(alpha_clipped_fragment.clone()),
                primitive: normal_primitive_state,
                depth_stencil: Some(normal_depth_state),
                multisample: Default::default(),
                multiview,
            }),
            opaque_mirrored: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("opaque mirrored pipeline"),
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
                    label: Some("alpha clipped pipeline"),
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
