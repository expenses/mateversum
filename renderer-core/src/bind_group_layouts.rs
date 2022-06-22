pub struct BindGroupLayouts {
    pub uniform: wgpu::BindGroupLayout,
    pub model: wgpu::BindGroupLayout,
    pub mirror_uniform: wgpu::BindGroupLayout,
    pub tonemap: wgpu::BindGroupLayout,
    pub ui_texture: wgpu::BindGroupLayout,
    pub skybox: wgpu::BindGroupLayout,
}

impl BindGroupLayouts {
    pub fn new(device: &wgpu::Device, options: &crate::pipelines::PipelineOptions) -> Self {
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

        Self {
            uniform: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniform bind group layout"),
                entries: &[
                    uniform_entry(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
                    sampler_entry(1),
                    texture_entry(2),
                    cubemap_entry(3),
                    cubemap_entry(4),
                ],
            }),
            mirror_uniform: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mirror bind group layout"),
                entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX)],
            }),
            model: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            tonemap: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mirror bind group layout"),
                entries: &[
                    sampler_entry(0),
                    if options.multiview.is_none() {
                        texture_entry(1)
                    } else {
                        texture_array_entry(1)
                    },
                ],
            }),
            ui_texture: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ui texture bind group layout"),
                entries: &[texture_entry(0)],
            }),
            skybox: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("skybox bind group layout"),
                entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX)],
            }),
        }
    }
}
