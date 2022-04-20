use glam::{Vec2, Vec3};
use kiss_engine_wgpu::{BindingResource, Device, RenderPipeline, Resource, Texture};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct ModelLoadContext<'a> {
    pub device: &'a Device,
    pub queue: &'a wgpu::Queue,
    pub url: url::Url,
    pub fetched_images: &'a mut FetchedImages,
    pub pbr_alpha_clipped_pipeline: &'a RenderPipeline,
    pub pbr_pipeline: &'a RenderPipeline,
    pub black_image: Arc<Resource<Texture>>,
    pub white_image: Arc<Resource<Texture>>,
    pub flat_normals_image: Arc<Resource<Texture>>,
    pub default_metallic_roughness_image: Arc<Resource<Texture>>,
}

struct ModelBuffers<'a> {
    map: std::collections::HashMap<usize, Vec<u8>>,
    blob: Option<&'a Vec<u8>>,
}

struct MaterialTextures {
    normal_texture: Arc<Resource<Texture>>,
    albedo_texture: Arc<Resource<Texture>>,
    metallic_roughness_texture: Arc<Resource<Texture>>,
    emissive_texture: Arc<Resource<Texture>>,
}

pub struct ModelPrimitive {
    pub indices: wgpu::Buffer,
    pub positions: wgpu::Buffer,
    pub normals: wgpu::Buffer,
    pub uvs: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub num_indices: u32,
    // We hold handles onto the used textures here, so that when the model is dropped, the `Arc::strong_count`
    // of the textures goes down. Then we are able to unload the textures from GPU memory by `HashMap::retain`ing the fetched images..
    _textures: MaterialTextures,
}

struct StagingModelPrimitive {
    indices: Vec<u32>,
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    textures: MaterialTextures,
}

impl StagingModelPrimitive {
    fn upload(self, context: &ModelLoadContext, alpha_clipped: bool) -> ModelPrimitive {
        ModelPrimitive {
            bind_group: context.device.create_owned_bind_group(
                None,
                if alpha_clipped {
                    context.pbr_alpha_clipped_pipeline
                } else {
                    context.pbr_pipeline
                },
                1,
                &[
                    BindingResource::Texture(&self.textures.albedo_texture),
                    BindingResource::Texture(&self.textures.normal_texture),
                    BindingResource::Texture(&self.textures.metallic_roughness_texture),
                    BindingResource::Texture(&self.textures.emissive_texture),
                ],
            ),
            num_indices: self.indices.len() as u32,
            indices: context
                .device
                .inner
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("indices"),
                    contents: bytemuck::cast_slice(&self.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
            positions: context
                .device
                .inner
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("positions"),
                    contents: bytemuck::cast_slice(&self.positions),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            normals: context
                .device
                .inner
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("normals"),
                    contents: bytemuck::cast_slice(&self.normals),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            uvs: context
                .device
                .inner
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("uvs"),
                    contents: bytemuck::cast_slice(&self.uvs),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            _textures: self.textures,
        }
    }
}

pub struct Model {
    pub opaque_primitives: Vec<ModelPrimitive>,
    pub alpha_clipped_primitives: Vec<ModelPrimitive>,
}

pub async fn load_gltf(context: &mut ModelLoadContext<'_>) -> Model {
    let bytes = fetch_bytes(&context.url).await;

    let gltf = gltf::Gltf::from_slice(&bytes).unwrap();

    let mut buffers = ModelBuffers {
        blob: gltf.blob.as_ref(),
        map: Default::default(),
    };

    let node_tree = gltf_helpers::NodeTree::new(gltf.nodes());

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
                    buffers
                        .map
                        .insert(buffer.index(), base64::decode(data).unwrap());
                } else {
                    buffers.map.insert(buffer.index(), fetch_bytes(&url).await);
                }
            }
        }
    }

    let mut opaque_primitives = std::collections::HashMap::new();
    let mut alpha_clipped_primitives = std::collections::HashMap::new();

    for (node, mesh) in gltf
        .nodes()
        .filter_map(|node| node.mesh().map(|mesh| (node, mesh)))
    {
        let transform = node_tree.transform_of(node.index());

        for primitive in mesh.primitives() {
            let material = primitive.material();

            let primitive_map = match material.alpha_mode() {
                gltf::material::AlphaMode::Opaque => &mut opaque_primitives,
                _ => &mut alpha_clipped_primitives,
            };

            // We can't use `or_insert_with` here as that uses a closure and closures aren't async.
            let staging_primitive = match primitive_map.entry(material.index()) {
                std::collections::hash_map::Entry::Occupied(occupied) => occupied.into_mut(),
                std::collections::hash_map::Entry::Vacant(vacancy) => {
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

                    vacancy.insert(StagingModelPrimitive {
                        indices: Default::default(),
                        positions: Default::default(),
                        normals: Default::default(),
                        uvs: Default::default(),
                        textures: MaterialTextures {
                            albedo_texture: if let Some(albedo_texture) = pbr.base_color_texture() {
                                load_image_from_gltf(
                                    &albedo_texture.texture(),
                                    true,
                                    &buffers,
                                    context,
                                )
                                .await
                            } else {
                                context.white_image.clone()
                            },
                            normal_texture: if let Some(normal_texture) = material.normal_texture()
                            {
                                load_image_from_gltf(
                                    &normal_texture.texture(),
                                    false,
                                    &buffers,
                                    context,
                                )
                                .await
                            } else {
                                context.flat_normals_image.clone()
                            },
                            metallic_roughness_texture: if let Some(metallic_roughness_texture) =
                                pbr.metallic_roughness_texture()
                            {
                                load_image_from_gltf(
                                    &metallic_roughness_texture.texture(),
                                    false,
                                    &buffers,
                                    context,
                                )
                                .await
                            } else {
                                context.default_metallic_roughness_image.clone()
                            },
                            emissive_texture: if let Some(emissive_texture) =
                                material.emissive_texture()
                            {
                                load_image_from_gltf(
                                    &emissive_texture.texture(),
                                    true,
                                    &buffers,
                                    context,
                                )
                                .await
                            } else {
                                context.black_image.clone()
                            },
                        },
                    })
                }
            };

            let reader = primitive.reader(|buffer| match buffer.source() {
                gltf::buffer::Source::Bin => Some(buffers.blob.unwrap()),
                gltf::buffer::Source::Uri(_) => {
                    buffers.map.get(&buffer.index()).map(|vec| &vec[..])
                }
            });

            staging_primitive.indices.extend(
                reader
                    .read_indices()
                    .unwrap()
                    .into_u32()
                    .map(|index| staging_primitive.positions.len() as u32 + index),
            );
            staging_primitive.positions.extend(
                reader
                    .read_positions()
                    .unwrap()
                    .map(|pos| transform * Vec3::from(pos)),
            );
            staging_primitive.normals.extend(
                reader
                    .read_normals()
                    .unwrap()
                    .map(|rot| transform.rotation * Vec3::from(rot)),
            );
            staging_primitive.uvs.extend(
                reader
                    .read_tex_coords(0)
                    .unwrap()
                    .into_f32()
                    .map(glam::Vec2::from),
            );
        }
    }

    Model {
        opaque_primitives: opaque_primitives
            .into_values()
            .map(|primitive| primitive.upload(context, false))
            .collect(),
        alpha_clipped_primitives: alpha_clipped_primitives
            .into_values()
            .map(|primitive| primitive.upload(context, true))
            .collect(),
    }
}

async fn load_image_from_gltf(
    texture: &gltf::Texture<'_>,
    srgb: bool,
    buffers: &ModelBuffers<'_>,
    context: &mut ModelLoadContext<'_>,
) -> Arc<Resource<Texture>> {
    let image = texture.source();

    match image.source() {
        gltf::image::Source::View { view, mime_type } => {
            log::info!("{} {}", texture.index(), mime_type);
            let buffer = view.buffer();

            let buffer = match buffer.source() {
                gltf::buffer::Source::Bin => buffers.blob.unwrap(),
                gltf::buffer::Source::Uri(_) => buffers
                    .map
                    .get(&buffer.index())
                    .map(|vec| &vec[..])
                    .unwrap(),
            };

            let bytes = &buffer[view.offset()..view.offset() + view.length()];

            let image = if mime_type == "image/ktx2" {
                load_ktx2(context.device, context.queue, bytes)
            } else {
                load_standard_image_format(context.device, bytes, srgb)
            };

            Arc::new(image)
        }
        gltf::image::Source::Uri { uri, mime_type } => {
            let url = url::Url::options()
                .base_url(Some(&context.url))
                .parse(uri)
                .unwrap();

            if url.scheme() == "data" {
                let (_mime_type, data) = url.path().split_once(',').unwrap();

                Arc::new(load_standard_image_format(
                    context.device,
                    &base64::decode(data).unwrap(),
                    srgb,
                ))
            } else {
                if let Some((image, cached_srgb)) = context.fetched_images.get(&url) {
                    if *cached_srgb == srgb {
                        return image.clone();
                    } else {
                        log::warn!(
                            "Same URL image is used twice, in both srgb and non-srgb formats: {}",
                            url
                        );
                    }
                }

                let bytes = fetch_bytes(&url).await;

                let image = if mime_type == Some("image/ktx2") {
                    load_ktx2(context.device, context.queue, &bytes)
                } else {
                    load_standard_image_format(context.device, &bytes, srgb)
                };

                let image = Arc::new(image);

                context.fetched_images.insert(url, (image.clone(), srgb));

                image
            }
        }
    }
}

fn load_ktx2(device: &Device, queue: &wgpu::Queue, bytes: &[u8]) -> Resource<Texture> {
    let ktx2 = ktx2::Reader::new(bytes).unwrap();
    let header = ktx2.header();
    let mut levels = Vec::new();

    for level in ktx2.levels() {
        match header.supercompression_scheme {
            Some(ktx2::SupercompressionScheme::Zstandard) => {
                use std::io::Read;
                let mut cursor = std::io::Cursor::new(level);
                let mut decoded = Vec::new();
                ruzstd::StreamingDecoder::new(&mut cursor)
                    .unwrap()
                    .read_to_end(&mut decoded)
                    .unwrap();
                levels.push(std::borrow::Cow::Owned(decoded));
            }
            Some(other) => panic!("Unsupported: {:?}", other),
            None => {
                levels.push(std::borrow::Cow::Borrowed(level));
            }
        }
    }

    //let flattened: Vec<u8> = levels.iter().flat_map(|vec| vec.iter().cloned()).collect();

    for dfd in ktx2.data_format_descriptors() {
        if dfd.header == ktx2::DataFormatDescriptorHeader::BASIC {
            let basic_data_format_descriptor =
                ktx2::BasicDataFormatDescriptor::parse(dfd.data).unwrap();

            log::info!("{:?}", basic_data_format_descriptor.color_model);
        }
    }

    log::info!("{:?} {:?}", levels.len(), header);

    todo!();
}

fn load_standard_image_format(
    device: &Device,
    format_bytes: &[u8],
    srgb: bool,
) -> Resource<Texture> {
    let image = image::load_from_memory(format_bytes).unwrap();

    let image = image.to_rgba8();

    let format = if srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };

    let internal_format = if srgb {
        glow::SRGB8_ALPHA8
    } else {
        glow::RGBA8
    };

    let mip_level_count = mip_levels_for_image_size(image.width(), image.height());

    let texture_descriptor = &wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: image.width(),
            height: image.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
    };

    // This is unfortunately the only way I could get mipmaps working as you can only write
    // to the 0th mip.
    let texture = unsafe {
        device.inner.as_hal::<wgpu_hal::gles::Api, _, _>(|device| {
            let device = device.unwrap();
            let gl_context = device.gl_context();

            use glow::HasContext;

            let texture = gl_context.create_texture().unwrap();
            gl_context.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl_context.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                internal_format as i32,
                image.width() as i32,
                image.height() as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                Some(&*image),
            );
            gl_context.generate_mipmap(glow::TEXTURE_2D);
            gl_context.bind_texture(glow::TEXTURE_2D, None);
            texture
        })
    };

    device.create_owned_texture_resource(unsafe {
        device.inner.create_texture_from_hal::<wgpu_hal::gles::Api>(
            wgpu_hal::gles::Texture {
                inner: wgpu_hal::gles::TextureInner::Texture {
                    raw: texture,
                    target: glow::TEXTURE_2D,
                },
                mip_level_count: 1,
                array_layer_count: 1,
                format,
                format_desc: wgpu_hal::gles::TextureFormatDesc {
                    internal: internal_format,
                    external: glow::RGBA,
                    data_type: glow::UNSIGNED_BYTE,
                },
                copy_size: wgpu_hal::CopyExtent {
                    width: image.width(),
                    height: image.height(),
                    depth: 1,
                },
            },
            texture_descriptor,
        )
    })
}

pub fn load_single_pixel_image(
    device: &Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    bytes: &[u8; 4],
) -> Arc<Resource<Texture>> {
    Arc::new(
        device.create_owned_texture_resource(device.inner.create_texture_with_data(
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
        )),
    )
}

async fn fetch_bytes(url: &url::Url) -> Vec<u8> {
    let response: web_sys::Response = wasm_bindgen_futures::JsFuture::from(
        web_sys::window().unwrap().fetch_with_str(url.as_str()),
    )
    .await
    .unwrap()
    .into();

    let length = response.headers().get("content-length").unwrap().unwrap();
    let length: u64 = length.parse().unwrap();

    log::info!("Size in MB: {}", length as f32 / 1024.0 / 1024.0);

    let array_buffer: js_sys::ArrayBuffer =
        wasm_bindgen_futures::JsFuture::from(response.array_buffer().unwrap())
            .await
            .unwrap()
            .into();

    let uint8_buffer = js_sys::Uint8Array::new(&array_buffer);

    uint8_buffer.to_vec()
}

pub type FetchedImages = std::collections::HashMap<url::Url, (Arc<Resource<Texture>>, bool)>;

pub fn prune_fetched_images(fetched_images: &mut FetchedImages) -> u32 {
    let mut removed = 0;

    fetched_images.retain(|_, (texture_ref, _)| {
        // Check the strong count. If a model is using the image then
        // it should be 2: the model + the one in this map. If less than 2
        // (it's normally impossible for strong_count to return 1 but w/e)
        // then we can drop it.
        if Arc::strong_count(texture_ref) < 2 {
            removed += 1;
            false
        } else {
            true
        }
    });

    removed
}

fn mip_levels_for_image_size(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2() as u32 + 1
}

/*
// Like the following, except without trying to write subsequent mips.
// https://github.com/gfx-rs/wgpu/blob/0b61a191244da0f0d987d53614a6698097a7622f/wgpu/src/util/device.rs#L79-L146
fn create_texture_with_first_mip_data(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    desc: &wgpu::TextureDescriptor,
    data: &[u8],
) -> wgpu::Texture {
    use std::num::NonZeroU32;

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
*/
