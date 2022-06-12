use crate::ResourceCache;

pub(crate) struct PipelineSet {
    pub(crate) opaque: wgpu::RenderPipeline,
    pub(crate) alpha_clipped: wgpu::RenderPipeline,
    pub(crate) opaque_mirrored: wgpu::RenderPipeline,
    pub(crate) alpha_clipped_mirrored: wgpu::RenderPipeline,
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
    ) -> Self {
        let normal_primitive_state = wgpu::PrimitiveState {
            front_face: wgpu::FrontFace::Ccw,
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
            front_face: wgpu::FrontFace::Cw,
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

pub(crate) struct Pipelines {
    pub(crate) pbr: PipelineSet,
    pub(crate) pbr_double_sided: PipelineSet,
    pub(crate) stencil_write: wgpu::RenderPipeline,
    pub(crate) set_depth: wgpu::RenderPipeline,
    pub(crate) tonemap: wgpu::RenderPipeline,
    pub(crate) ui: wgpu::RenderPipeline,
    pub(crate) skybox: wgpu::RenderPipeline,
    pub(crate) skybox_mirrored: wgpu::RenderPipeline,
}

impl Pipelines {
    pub fn new(
        device: &wgpu::Device,
        shader_cache: &ResourceCache<wgpu::ShaderModule>,
        uniform_bgl: &wgpu::BindGroupLayout,
        model_bgl: &wgpu::BindGroupLayout,
        mirror_uniform_bgl: &wgpu::BindGroupLayout,
        tonemap_bgl: &wgpu::BindGroupLayout,
        ui_texture_bgl: &wgpu::BindGroupLayout,
        skybox_bgl: &wgpu::BindGroupLayout,
        multiview: Option<std::num::NonZeroU32>,
    ) -> Self {
        let uniform_only_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("uniform only pipeline layout"),
                bind_group_layouts: &[uniform_bgl],
                push_constant_ranges: &[],
            });

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

        let prefix = if multiview.is_none() {
            "single_view::"
        } else {
            ""
        };

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
                layout: Some(&uniform_only_pipeline_layout),
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
            layout: Some(&uniform_only_pipeline_layout),
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

        let ui_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui pipeline layout"),
            bind_group_layouts: &[uniform_bgl, ui_texture_bgl],
            push_constant_ranges: &[],
        });

        let skybox_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("skybox pipeline layout"),
                bind_group_layouts: &[uniform_bgl, skybox_bgl],
                push_constant_ranges: &[],
            });

        let skybox_mirrored_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("skybox mirrored pipeline layout"),
                bind_group_layouts: &[uniform_bgl, skybox_bgl, mirror_uniform_bgl],
                push_constant_ranges: &[],
            });

        let fragment_opaque = wgpu::FragmentState {
            module: shader_cache.get("fragment", || {
                device.create_shader_module(&if multiview.is_none() {
                    wgpu::include_spirv!("../compiled-shaders/single_view_fragment.spv")
                } else {
                    wgpu::include_spirv!("../compiled-shaders/fragment.spv")
                })
            }),
            entry_point: &format!("{}fragment", prefix),
            targets: &[wgpu::TextureFormat::Rgba16Float.into()],
        };

        let fragment_alpha_clipped = wgpu::FragmentState {
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
        };

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
            ),
            stencil_write: stencil_write_pipeline,
            set_depth: set_depth_pipeline,
            tonemap: tonemap_pipeline,
            ui: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ui pipeline"),
                layout: Some(&ui_pipeline_layout),
                vertex: vertex_state.clone(),
                fragment: Some(wgpu::FragmentState {
                    module: shader_cache.get("fragment_ui", || {
                        device.create_shader_module(&wgpu::include_spirv!(
                            "../compiled-shaders/fragment_ui.spv"
                        ))
                    }),
                    entry_point: "fragment_ui",
                    targets: &[wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
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
                    module: shader_cache.get("vertex_skybox", || {
                        device.create_shader_module(&if multiview.is_none() {
                            wgpu::include_spirv!(
                                "../compiled-shaders/single_view_vertex_skybox.spv"
                            )
                        } else {
                            wgpu::include_spirv!("../compiled-shaders/vertex_skybox.spv")
                        })
                    }),
                    entry_point: &format!("{}vertex_skybox", prefix),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader_cache.get("fragment_skybox", || {
                        device.create_shader_module(&wgpu::include_spirv!(
                            "../compiled-shaders/fragment_skybox.spv"
                        ))
                    }),
                    entry_point: "fragment_skybox",
                    targets: &[wgpu::TextureFormat::Rgba16Float.into()],
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
                    module: shader_cache.get("vertex_skybox_mirrored", || {
                        device.create_shader_module(&if multiview.is_none() {
                            wgpu::include_spirv!(
                                "../compiled-shaders/single_view_vertex_skybox_mirrored.spv"
                            )
                        } else {
                            wgpu::include_spirv!("../compiled-shaders/vertex_skybox_mirrored.spv")
                        })
                    }),
                    entry_point: &format!("{}vertex_skybox_mirrored", prefix),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader_cache.get("fragment_skybox", || {
                        device.create_shader_module(&wgpu::include_spirv!(
                            "../compiled-shaders/fragment_skybox.spv"
                        ))
                    }),
                    entry_point: "fragment_skybox",
                    targets: &[wgpu::TextureFormat::Rgba16Float.into()],
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
        }
    }
}
