use super::*;

// Like the following, except without trying to write subsequent mips.
// https://github.com/gfx-rs/wgpu/blob/0b61a191244da0f0d987d53614a6698097a7622f/wgpu/src/util/device.rs#L79-L146
pub(super) fn create_texture_with_first_mip_data(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    desc: &wgpu::TextureDescriptor,
    data: &[u8],
) -> wgpu::Texture {
    // Implicitly add the COPY_DST usage
    let mut desc = desc.to_owned();
    desc.usage |= wgpu::TextureUsages::COPY_DST;
    let texture = device.create_texture(&desc);

    let format_info = desc.format.describe();
    let layer_iterations = desc.array_layer_count();

    let mut binary_offset = 0;
    for layer in 0..layer_iterations {
        let width_blocks = desc.size.width / format_info.block_dimensions.0 as u32;
        let height_blocks = desc.size.height / format_info.block_dimensions.1 as u32;

        let bytes_per_row = width_blocks * format_info.block_size as u32;
        let data_size = bytes_per_row * height_blocks * desc.size.depth_or_array_layers;

        let end_offset = binary_offset + data_size as usize;

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &data[binary_offset..end_offset],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(bytes_per_row).expect("invalid bytes per row")),
                rows_per_image: Some(NonZeroU32::new(height_blocks).expect("invalid height")),
            },
            desc.size,
        );

        binary_offset = end_offset;
    }

    texture
}

pub(super) fn load_standard_image_format(
    context: &ModelLoadContext,
    format_bytes: &[u8],
    srgb: bool,
) -> Texture {
    let image = image::load_from_memory(format_bytes).unwrap();

    let image = image.to_rgba8();

    let mip_level_count = mip_levels_for_image_size(image.width(), image.height());

    let format = if srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };

    let texture = Texture::new(create_texture_with_first_mip_data(
        &context.device,
        &context.queue,
        &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: image.width(),
                height: image.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if srgb {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
        },
        &*image,
    ));

    let temp_blit_textures: Vec<_> = (1..mip_level_count)
        .map(|level| {
            let mip_extent = wgpu::Extent3d {
                width: image.width() >> level,
                height: image.height() >> level,
                depth_or_array_layers: 1,
            };

            Texture::new(context.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: mip_extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
            }))
        })
        .collect();

    let source_view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
        mip_level_count: Some(std::num::NonZeroU32::new(1).unwrap()),
        ..Default::default()
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("command encoder"),
        });

    let pipeline = context.pipeline_cache.get(
        if srgb {
            "blit pipeline (srgb)"
        } else {
            "blit pipeline"
        },
        || {
            let bind_group_layout =
                context
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                count: None,
                                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                count: None,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: true,
                                    },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                            },
                        ],
                    });

            let pipeline_layout =
                context
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let pipeline = context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: context.shader_cache.get("fullscreen_tri", || {
                            context.device.create_shader_module(&wgpu::include_spirv!(
                                "../../fullscreen_tri.spv"
                            ))
                        }),
                        entry_point: "fullscreen_tri",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: context.shader_cache.get("blit", || {
                            context
                                .device
                                .create_shader_module(&wgpu::include_spirv!("../../blit.spv"))
                        }),
                        entry_point: "blit",
                        targets: &[format.into()],
                    }),
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    multiview: Default::default(),
                });

            PipelineData {
                pipeline,
                bind_group_layout,
                pipeline_layout,
            }
        },
    );

    for source_level in 0..mip_level_count - 1 {
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(&context.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(if source_level == 0 {
                            &source_view
                        } else {
                            &temp_blit_textures[source_level as usize - 1].view
                        }),
                    },
                ],
            });

        let temp_blit_texture = &temp_blit_textures[source_level as usize];

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit render pass"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &temp_blit_texture.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        render_pass.set_pipeline(&pipeline.pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);

        drop(render_pass);

        let target_level = source_level + 1;

        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &temp_blit_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &texture.texture,
                mip_level: source_level + 1,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: image.width() >> target_level,
                height: image.height() >> target_level,
                depth_or_array_layers: 1,
            },
        );
    }

    context.queue.submit(std::iter::once(encoder.finish()));

    texture
}

pub(super) fn load_ktx2_sync(
    device: &Rc<wgpu::Device>,
    queue: &wgpu::Queue,
    srgb: bool,
    format: Format,
    bytes: &[u8],
) -> Texture {
    let ktx2 = ktx2::Reader::new(bytes).unwrap();
    let header = ktx2.header();

    for dfd in ktx2.data_format_descriptors() {
        if dfd.header == ktx2::DataFormatDescriptorHeader::BASIC {
            let basic_dfd = ktx2::BasicDataFormatDescriptor::parse(dfd.data).unwrap();
            let sample_information: Vec<_> = basic_dfd.sample_information().collect();
            log::info!("{:?} {:?}", basic_dfd.color_model, sample_information);
        }
    }

    let transcoder = basis_universal::LowLevelUastcTranscoder::new();

    let texture_descriptor = &wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: header.pixel_width,
            height: header.pixel_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: header.level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: format.as_wgpu(srgb),
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    };

    let texture = device.create_texture(texture_descriptor);

    let (block_width_pixels, block_height_pixels) = (4, 4);

    for (i, level) in ktx2.levels().enumerate() {
        let decompressed = match header.supercompression_scheme {
            Some(ktx2::SupercompressionScheme::Zstandard) => {
                let decompressed =
                    zstd::bulk::decompress(level.bytes, level.uncompressed_length_bytes as usize)
                        .unwrap();
                std::borrow::Cow::Owned(decompressed)
            }
            Some(other) => panic!("Unsupported: {:?}", other),
            None => std::borrow::Cow::Borrowed(level.bytes),
        };

        let slice_width = header.pixel_width >> i;
        let slice_height = header.pixel_height >> i;

        let slice_parameters = basis_universal::SliceParametersUastc {
            num_blocks_x: ((slice_width + block_width_pixels - 1) / block_width_pixels).max(1),
            num_blocks_y: ((slice_height + block_height_pixels - 1) / block_height_pixels).max(1),
            has_alpha: false,
            original_width: slice_width,
            original_height: slice_height,
        };

        let transcoded = transcoder
            .transcode_slice(
                &decompressed,
                slice_parameters,
                basis_universal::DecodeFlags::HIGH_QUALITY,
                format.as_transcoder_block_format(),
            )
            .unwrap();

        write_bytes_to_texture(queue, &texture, i as u32, &transcoded, texture_descriptor);
    }

    Texture::new(texture)
}

pub(super) fn load_basis(
    device: &Rc<wgpu::Device>,
    queue: &wgpu::Queue,
    format: Format,
    bytes: &[u8],
    srgb: bool,
) -> Texture {
    let array = unsafe { js_sys::Uint8Array::view(bytes) };

    let file = basis_universal_wasm::BasisFile::new(&array);

    let image = 0;

    let num_levels = file.get_num_levels(image);

    let total_transcoded_size: u32 = (0..num_levels)
        .map(|level| {
            file.get_image_transcoded_size_in_bytes(
                image,
                level,
                format.as_transcoder_format() as u32,
            )
        })
        .sum();

    let transcoded_data = vec![0; total_transcoded_size as usize];

    assert_eq!(file.start_transcoding(), 1);

    let mut offset = 0;

    for level in 0..num_levels {
        let size = file.get_image_transcoded_size_in_bytes(
            image,
            level,
            format.as_transcoder_format() as u32,
        );

        let slice = unsafe {
            js_sys::Uint8Array::view(
                &transcoded_data[offset as usize..offset as usize + size as usize],
            )
        };

        offset += size;

        let res = file.transcode_image(
            &slice,
            image,
            level as u32,
            format.as_transcoder_format() as u32,
            1,
            0,
        );

        assert_eq!(res, 1);
    }

    let width = file.get_image_width(image, 0);
    let height = file.get_image_height(image, 0);

    file.close();
    file.delete();

    let format = format.as_wgpu(srgb);

    Texture::new(device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: num_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
        },
        &transcoded_data,
    ))
}
