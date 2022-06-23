
use std::sync::Arc;
use crate::Texture;
use std::num::NonZeroU32;
use wgpu::util::DeviceExt;
use std::io::Read;
use super::HttpClient;

pub struct Context<T: HttpClient + 'static> {
    pub pipelines: Arc<crate::Pipelines>,
    pub bind_group_layouts: Arc<crate::BindGroupLayouts>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub http_client: T,
}

pub async fn load_ktx2_cubemap<T: HttpClient + Clone + 'static>(
    context: &Context<T>,
    url: &url::Url,
) -> anyhow::Result<Arc<Texture>> {
    let mut header_bytes = [0; ktx2::Header::LENGTH];

    let fetched_header = context
        .http_client
        .fetch_bytes(url, Some(0..ktx2::Header::LENGTH))
        .await?;

    header_bytes.copy_from_slice(&fetched_header);

    let header = ktx2::Header::from_bytes(&header_bytes)?;

    if header.face_count != 6 {
        return Err(anyhow::anyhow!(
            "Expected 6 faces, got {}",
            header.face_count
        ));
    }

    if header.format != Some(ktx2::Format::BC6H_UFLOAT_BLOCK) {
        return Err(anyhow::anyhow!(
            "Got an unsupported format: {:?}",
            header.format
        ));
    }

    if header.supercompression_scheme != Some(ktx2::SupercompressionScheme::Zstandard) {
        return Err(anyhow::anyhow!(
            "Got an unsupported supercompression scheme: {:?}",
            header.supercompression_scheme
        ));
    }

    let mut level_indices = Vec::with_capacity(header.level_count as usize);

    {
        let mut reader = std::io::Cursor::new(
            context
                .http_client
                .fetch_bytes(
                    url,
                    Some(
                        ktx2::Header::LENGTH
                            ..ktx2::Header::LENGTH
                                + ktx2::LevelIndex::LENGTH * header.level_count as usize,
                    ),
                )
                .await?,
        );

        for _ in 0..header.level_count {
            let mut level_index_bytes = [0; ktx2::LevelIndex::LENGTH];

            reader.read_exact(&mut level_index_bytes)?;

            level_indices.push(ktx2::LevelIndex::from_bytes(&level_index_bytes));
        }
    }

    // Compressed textures made made of 4x4 blocks, so there are some issues
    // with textures that don't have a side length divisible by 4.
    // They're considered fine everywhere except D3D11 and old versions of D3D12
    // (according to jasperrlz in the Wgpu Users element chat).
    let base_width = header.pixel_width - (header.pixel_width % 4);
    let base_height = header.pixel_height - (header.pixel_height % 4);

    let bc6h_supported = context.device.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC);

    let texture_descriptor = move || wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: base_width,
            height: base_height,
            depth_or_array_layers: 6,
        },
        mip_level_count: header.level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: if bc6h_supported {
            wgpu::TextureFormat::Bc6hRgbUfloat
        } else {
            wgpu::TextureFormat::Rg11b10Float
        },
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    };

    let texture = context.device.create_texture(&texture_descriptor());

    let texture = Arc::new(Texture {
        view: texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        }),
        texture,
    });

    wasm_bindgen_futures::spawn_local({
        let texture = Arc::clone(&texture);
        let device = Arc::clone(&context.device);
        let queue = Arc::clone(&context.queue);
        let pipelines = Arc::clone(&context.pipelines);
        let bind_group_layouts = Arc::clone(&context.bind_group_layouts);

        let http_client = context.http_client.clone();
        let url = url.clone();

        async move {
            for (i, level_index) in level_indices.into_iter().enumerate().rev() {
                let bytes = http_client
                    .fetch_bytes(
                        &url,
                        Some(
                            level_index.byte_offset as usize
                                ..(level_index.byte_offset + level_index.byte_length) as usize,
                        ),
                    )
                    .await
                    .unwrap();

                let decompressed = match header.supercompression_scheme {
                    Some(ktx2::SupercompressionScheme::Zstandard) => zstd::bulk::decompress(
                        &bytes,
                        level_index.uncompressed_byte_length as usize,
                    )
                    .unwrap(),
                    Some(other) => panic!("Unsupported: {:?}", other),
                    None => bytes.to_vec(),
                };

                if !bc6h_supported {
                    let mut command_encoder =
                        device.create_command_encoder(&Default::default());

                    let stride = decompressed.len() / 6;

                    for face in 0..6 {
                        let bytes = &decompressed[face * stride..(face + 1) * stride];

                        let input_texture = device.create_texture_with_data(
                            &queue,
                            &wgpu::TextureDescriptor {
                                label: None,
                                size: wgpu::Extent3d {
                                    width: base_width >> (i + 2),
                                    height: base_height >> (i + 2),
                                    depth_or_array_layers: 1,
                                },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D2,
                                format: wgpu::TextureFormat::Rgba32Uint,
                                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                            },
                            bytes,
                        );

                        let output_texture =
                            device.create_texture(&wgpu::TextureDescriptor {
                                label: None,
                                size: wgpu::Extent3d {
                                    width: base_width >> i,
                                    height: base_height >> i,
                                    depth_or_array_layers: 1,
                                },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D2,
                                format: wgpu::TextureFormat::Rg11b10Float,
                                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                                    | wgpu::TextureUsages::COPY_SRC,
                            });

                        let bind_group =
                            device
                                .create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: None,
                                    layout: &bind_group_layouts.uint_texture,
                                    entries: &[wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(
                                            &input_texture.create_view(&Default::default()),
                                        ),
                                    }],
                                });

                        let output_view = output_texture.create_view(&Default::default());

                        let mut render_pass =
                            command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: None,
                                color_attachments: &[wgpu::RenderPassColorAttachment {
                                    view: &output_view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: true,
                                    },
                                }],
                                depth_stencil_attachment: None,
                            });

                        render_pass.set_pipeline(&pipelines.bc6h_decompression);

                        render_pass.set_bind_group(0, &bind_group, &[]);

                        render_pass.draw(0..3, 0..1);

                        drop(render_pass);

                        command_encoder.copy_texture_to_texture(
                            wgpu::ImageCopyTexture {
                                texture: &output_texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::ImageCopyTexture {
                                texture: &texture.texture,
                                mip_level: i as u32,
                                origin: wgpu::Origin3d {
                                    x: 0,
                                    y: 0,
                                    z: face as u32,
                                },
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::Extent3d {
                                width: base_width >> i,
                                height: base_width >> i,
                                depth_or_array_layers: 1,
                            },
                        );
                    }

                    queue
                        .submit(std::iter::once(command_encoder.finish()));
                } else {
                    write_bytes_to_texture(
                        &queue,
                        &texture.texture,
                        i as u32,
                        &decompressed,
                        &texture_descriptor(),
                    );
                }
            }
        }
    });

    Ok(texture)
}

fn write_bytes_to_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip: u32,
    bytes: &[u8],
    desc: &wgpu::TextureDescriptor,
) {
    let format_info = desc.format.describe();

    let mip_size = desc.mip_level_size(mip).unwrap();

    let mip_physical = mip_size.physical_size(desc.format);

    let width_blocks = mip_physical.width / format_info.block_dimensions.0 as u32;
    let height_blocks = mip_physical.height / format_info.block_dimensions.1 as u32;

    let bytes_per_row = width_blocks * format_info.block_size as u32;

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: mip,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(NonZeroU32::new(bytes_per_row).expect("invalid bytes per row")),
            rows_per_image: Some(NonZeroU32::new(height_blocks).expect("invalid height")),
        },
        mip_physical,
    );
}