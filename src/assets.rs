use crevice::std140::AsStd140;
use futures::AsyncReadExt;
use glam::{Vec2, Vec3};
use std::borrow::Cow;
use std::cell::RefCell;
use std::num::NonZeroU32;
use std::rc::Rc;
use wgpu::util::DeviceExt;

use crate::caching::{PipelineData, ResourceCache};

pub(crate) struct ModelLoadContext {
    pub(crate) device: Rc<wgpu::Device>,
    pub(crate) queue: Rc<wgpu::Queue>,
    pub(crate) fetched_images: Rc<RefCell<FetchedImages>>,
    pub(crate) model_bgl: Rc<wgpu::BindGroupLayout>,
    pub(crate) black_image: Rc<Texture>,
    pub(crate) white_image: Rc<Texture>,
    pub(crate) flat_normals_image: Rc<Texture>,
    pub(crate) default_metallic_roughness_image: Rc<Texture>,
    pub(crate) supported_features: wgpu::Features,
    pub(crate) shader_cache: Rc<ResourceCache<wgpu::ShaderModule>>,
    pub(crate) pipeline_cache: Rc<ResourceCache<PipelineData>>,
    pub(crate) sampler: Rc<wgpu::Sampler>,
    pub(crate) anisotropy_clamp: Option<std::num::NonZeroU8>,
}

struct ModelBuffers {
    map: std::collections::HashMap<usize, Vec<u8>>,
}

struct MaterialTexture {
    texture: Rc<RefCell<Rc<Texture>>>,
    sampler: Rc<RefCell<Rc<wgpu::Sampler>>>,
}

impl MaterialTexture {
    fn new(texture: Rc<Texture>, context: &ModelLoadContext) -> Self {
        Self {
            texture: Rc::new(RefCell::new(texture)),
            sampler: Rc::new(RefCell::new(Rc::clone(&context.sampler))),
        }
    }
}

struct MaterialTextures {
    normal_texture: MaterialTexture,
    albedo_texture: MaterialTexture,
    metallic_roughness_texture: MaterialTexture,
    emissive_texture: MaterialTexture,
}

struct TextureLoadContext {
    gltf: Rc<gltf::Gltf>,
    context: Rc<ModelLoadContext>,
    buffers: Rc<ModelBuffers>,
    textures: Rc<MaterialTextures>,
    material_settings: Rc<wgpu::Buffer>,
    bind_group: Rc<RefCell<wgpu::BindGroup>>,
    base_url: Rc<Option<url::Url>>,
}

async fn upload_model_texture_from_gltf(
    gltf_texture: &gltf::Texture<'_>,
    binding: &MaterialTexture,
    srgb: bool,
    context: &TextureLoadContext,
) -> anyhow::Result<()> {
    let texture = load_image_from_gltf(context, gltf_texture, srgb, binding).await?;

    *binding.texture.borrow_mut() = texture;

    let new_bind_group = create_model_bind_group(
        &context.context,
        &context.textures,
        &context.material_settings,
    );

    *context.bind_group.borrow_mut() = new_bind_group;

    Ok(())
}

fn create_model_bind_group(
    context: &ModelLoadContext,
    textures: &MaterialTextures,
    material_settings: &wgpu::Buffer,
) -> wgpu::BindGroup {
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &context.model_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &textures.albedo_texture.texture.borrow().view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &textures.normal_texture.texture.borrow().view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &textures.metallic_roughness_texture.texture.borrow().view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &textures.emissive_texture.texture.borrow().view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: material_settings.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(
                        &textures.albedo_texture.sampler.borrow(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(
                        &textures.normal_texture.sampler.borrow(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(
                        &textures.metallic_roughness_texture.sampler.borrow(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(
                        &textures.emissive_texture.sampler.borrow(),
                    ),
                },
            ],
        })
}

pub(crate) struct ModelPrimitive {
    pub(crate) indices: wgpu::Buffer,
    pub(crate) positions: wgpu::Buffer,
    pub(crate) normals: wgpu::Buffer,
    pub(crate) uvs: wgpu::Buffer,
    pub(crate) bind_group: Rc<RefCell<wgpu::BindGroup>>,
    pub(crate) num_indices: u32,
    // We hold handles onto the used textures here, so that when the model is dropped, the `Rc::strong_count`
    // of the textures goes down. Then we are able to unload the textures from GPU memory by `HashMap::retain`ing the fetched images..
    _textures: Rc<MaterialTextures>,
}

struct StagingModelPrimitive {
    indices: Vec<u32>,
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    material_index: usize,
    material_settings: Rc<wgpu::Buffer>,
}

impl StagingModelPrimitive {
    fn upload(
        self,
        gltf: &Rc<gltf::Gltf>,
        context: &Rc<ModelLoadContext>,
        buffers: &Rc<ModelBuffers>,
        base_url: &Rc<Option<url::Url>>,
    ) -> ModelPrimitive {
        let textures = Rc::new(MaterialTextures {
            albedo_texture: MaterialTexture::new(Rc::clone(&context.white_image), context),
            normal_texture: MaterialTexture::new(Rc::clone(&context.flat_normals_image), context),
            metallic_roughness_texture: MaterialTexture::new(
                Rc::clone(&context.default_metallic_roughness_image),
                context,
            ),
            emissive_texture: MaterialTexture::new(Rc::clone(&context.black_image), context),
        });

        let material_index = self.material_index;
        let material_settings = self.material_settings;

        let bind_group = Rc::new(RefCell::new(create_model_bind_group(
            context,
            &textures,
            &material_settings,
        )));

        let texture_load_context = Rc::new(TextureLoadContext {
            gltf: Rc::clone(gltf),
            context: Rc::clone(context),
            buffers: Rc::clone(buffers),
            textures: Rc::clone(&textures),
            material_settings: Rc::clone(&material_settings),
            bind_group: Rc::clone(&bind_group),
            base_url: Rc::clone(base_url),
        });

        wasm_bindgen_futures::spawn_local({
            let textures = Rc::clone(&textures);
            let context = Rc::clone(&texture_load_context);
            async move {
                let material = context.gltf.materials().nth(material_index).unwrap();
                let pbr = material.pbr_metallic_roughness();
                if let Some(albedo_texture) = pbr.base_color_texture() {
                    upload_model_texture_from_gltf(
                        &albedo_texture.texture(),
                        &textures.albedo_texture,
                        true,
                        &context,
                    )
                    .await
                    .unwrap();
                }
            }
        });

        wasm_bindgen_futures::spawn_local({
            let textures = Rc::clone(&textures);
            let context = Rc::clone(&texture_load_context);
            async move {
                let material = context.gltf.materials().nth(material_index).unwrap();
                if let Some(normal_texture) = material.normal_texture() {
                    upload_model_texture_from_gltf(
                        &normal_texture.texture(),
                        &textures.normal_texture,
                        false,
                        &context,
                    )
                    .await
                    .unwrap();
                }
            }
        });

        wasm_bindgen_futures::spawn_local({
            let textures = Rc::clone(&textures);
            let context = Rc::clone(&texture_load_context);
            async move {
                let material = context.gltf.materials().nth(material_index).unwrap();
                let pbr = material.pbr_metallic_roughness();
                if let Some(metallic_roughness_texture) = pbr.metallic_roughness_texture() {
                    upload_model_texture_from_gltf(
                        &metallic_roughness_texture.texture(),
                        &textures.metallic_roughness_texture,
                        false,
                        &context,
                    )
                    .await
                    .unwrap();
                }
            }
        });

        wasm_bindgen_futures::spawn_local({
            let textures = Rc::clone(&textures);
            let context = Rc::clone(&texture_load_context);
            async move {
                let material = context.gltf.materials().nth(material_index).unwrap();
                if let Some(emissive_texture) = material.emissive_texture() {
                    upload_model_texture_from_gltf(
                        &emissive_texture.texture(),
                        &textures.emissive_texture,
                        true,
                        &context,
                    )
                    .await
                    .unwrap();
                }
            }
        });

        ModelPrimitive {
            bind_group,
            num_indices: self.indices.len() as u32,
            indices: context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("indices"),
                    contents: bytemuck::cast_slice(&self.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
            positions: context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("positions"),
                    contents: bytemuck::cast_slice(&self.positions),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            normals: context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("normals"),
                    contents: bytemuck::cast_slice(&self.normals),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            uvs: context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("uvs"),
                    contents: bytemuck::cast_slice(&self.uvs),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            _textures: textures,
        }
    }
}

#[derive(Default)]
pub(crate) struct Model {
    pub(crate) opaque_primitives: Vec<ModelPrimitive>,
    pub(crate) alpha_clipped_primitives: Vec<ModelPrimitive>,
}

pub(crate) async fn load_gltf_from_bytes(
    bytes: &[u8],
    base_url: Option<url::Url>,
    context: Rc<ModelLoadContext>,
) -> anyhow::Result<Model> {
    let gltf = gltf::Gltf::from_slice(bytes).unwrap();

    let mut buffers = ModelBuffers {
        map: Default::default(),
    };

    let node_tree = gltf_helpers::NodeTree::new(gltf.nodes());

    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => {}
            gltf::buffer::Source::Uri(uri) => {
                let url = url::Url::options()
                    .base_url(base_url.as_ref())
                    .parse(uri)
                    .unwrap();

                if url.scheme() == "data" {
                    let (mime_type, data) = url.path().split_once(',').unwrap();
                    log::info!("Got: {}", mime_type);
                    buffers
                        .map
                        .insert(buffer.index(), base64::decode(data).unwrap());
                } else {
                    buffers.map.insert(buffer.index(), fetch_bytes(&url).await?);
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

                    vacancy.insert(StagingModelPrimitive {
                        indices: Default::default(),
                        positions: Default::default(),
                        normals: Default::default(),
                        uvs: Default::default(),
                        material_settings: Rc::new(
                            context
                                .device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("material settings"),
                                    contents: bytemuck::bytes_of(
                                        &shared_structs::MaterialSettings {
                                            base_color_factor: pbr.base_color_factor().into(),
                                            emissive_factor: material.emissive_factor().into(),
                                            metallic_factor: pbr.metallic_factor(),
                                            roughness_factor: pbr.roughness_factor(),
                                        }
                                        .as_std140(),
                                    ),
                                    usage: wgpu::BufferUsages::UNIFORM,
                                }),
                        ),
                        material_index: material.index().unwrap_or(0),
                    })
                }
            };

            let reader = primitive.reader(|buffer| match buffer.source() {
                gltf::buffer::Source::Bin => Some(gltf.blob.as_ref().unwrap()),
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

    let buffers = Rc::new(buffers);
    let gltf = Rc::new(gltf);
    let base_url = Rc::new(base_url);

    let mut opaque_primitives_vec = Vec::new();
    let mut alpha_clipped_primitives_vec = Vec::new();

    for (primitive, is_opaque) in opaque_primitives
        .into_values()
        .map(|primitive| (primitive, true))
        .chain(
            alpha_clipped_primitives
                .into_values()
                .map(|primitive| (primitive, false)),
        )
    {
        let primitive = primitive.upload(&gltf, &context, &buffers, &base_url);

        if is_opaque {
            opaque_primitives_vec.push(primitive);
        } else {
            alpha_clipped_primitives_vec.push(primitive);
        }
    }

    Ok(Model {
        opaque_primitives: opaque_primitives_vec,
        alpha_clipped_primitives: alpha_clipped_primitives_vec,
    })
}

async fn load_image_from_gltf(
    context: &TextureLoadContext,
    texture: &gltf::Texture<'_>,
    srgb: bool,
    binding: &MaterialTexture,
) -> anyhow::Result<Rc<Texture>> {
    let image = texture.source();

    Ok(match image.source() {
        gltf::image::Source::View { view, mime_type } => {
            log::info!("{} {}", texture.index(), mime_type);
            let buffer = view.buffer();

            let buffer = match buffer.source() {
                gltf::buffer::Source::Bin => context.gltf.blob.as_ref().unwrap(),
                gltf::buffer::Source::Uri(_) => context
                    .buffers
                    .map
                    .get(&buffer.index())
                    .map(|vec| &vec[..])
                    .unwrap(),
            };

            let bytes = &buffer[view.offset()..view.offset() + view.length()];

            load_image_from_mime_type(
                context,
                ImageSource::Bytes(bytes),
                srgb,
                Some(mime_type),
                binding,
            )
            .await?
        }
        gltf::image::Source::Uri { uri, mime_type } => {
            let url = url::Url::options()
                .base_url(context.base_url.as_ref().as_ref())
                .parse(uri)
                .unwrap();

            if url.scheme() == "data" {
                let (_mime_type, data) = url.path().split_once(',').unwrap();

                log::error!("Need to check mime type here. Does it indicate what file to load? Mime type: 1: {:?}, 2: {:?}", mime_type, _mime_type);

                Rc::new(load_standard_image_format(
                    &context.context,
                    &base64::decode(data).unwrap(),
                    srgb,
                ))
            } else {
                if let Some((image, cached_srgb)) =
                    context.context.fetched_images.borrow().get(&url)
                {
                    if *cached_srgb == srgb {
                        return Ok(Rc::clone(image));
                    } else {
                        log::warn!(
                            "Same URL image is used twice, in both srgb and non-srgb formats: {}",
                            url
                        );
                    }
                }

                let url = Rc::new(url);

                let image = load_image_from_mime_type(
                    context,
                    ImageSource::Url(Rc::clone(&url)),
                    srgb,
                    mime_type,
                    binding,
                )
                .await?;

                context
                    .context
                    .fetched_images
                    .borrow_mut()
                    .insert(url, (Rc::clone(&image), srgb));

                image
            }
        }
    })
}

enum ImageSource<'a> {
    Url(Rc<url::Url>),
    Bytes(&'a [u8]),
}

impl<'a> ImageSource<'a> {
    async fn get_bytes(&self) -> anyhow::Result<Cow<'a, [u8]>> {
        Ok(match self {
            Self::Url(url) => Cow::Owned(fetch_bytes(url).await?),
            Self::Bytes(bytes) => Cow::Borrowed(bytes),
        })
    }
}

async fn load_image_from_mime_type(
    context: &TextureLoadContext,
    source: ImageSource<'_>,
    srgb: bool,
    mime_type: Option<&str>,
    binding: &MaterialTexture,
) -> anyhow::Result<Rc<Texture>> {
    Ok(if mime_type == Some("image/ktx2") {
        match source {
            ImageSource::Bytes(bytes) => Rc::new(load_ktx2(
                &context.context.device,
                &context.context.queue,
                srgb,
                context.context.supported_features,
                bytes,
            )),
            ImageSource::Url(url) => load_ktx2_async(context, srgb, &url, binding).await?,
        }
    } else if mime_type == Some("image/x.basis") {
        Rc::new(load_basis(
            &context.context.device,
            &context.context.queue,
            context.context.supported_features,
            &source.get_bytes().await?,
            srgb,
        ))
    } else {
        Rc::new(load_standard_image_format(
            &context.context,
            &source.get_bytes().await?,
            srgb,
        ))
    })
}

#[derive(Clone, Copy, Debug)]
enum Format {
    Etc2Rgba,
    Bc7,
    Astc,
    Rgba,
}

impl Format {
    fn new_from_features(features: wgpu::Features) -> Self {
        if features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC_LDR) {
            Self::Astc
        } else if features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
            Self::Bc7
        } else if features.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2) {
            Self::Etc2Rgba
        } else {
            Self::Rgba
        }
    }

    fn as_transcoder_block_format(&self) -> basis_universal::TranscoderBlockFormat {
        match self {
            Self::Etc2Rgba => basis_universal::TranscoderBlockFormat::ETC2_RGBA,
            Self::Bc7 => basis_universal::TranscoderBlockFormat::BC7,
            Self::Astc => basis_universal::TranscoderBlockFormat::ASTC_4x4,
            Self::Rgba => basis_universal::TranscoderBlockFormat::RGBA32,
        }
    }

    fn as_transcoder_format(&self) -> basis_universal::TranscoderTextureFormat {
        match self {
            Self::Etc2Rgba => basis_universal::TranscoderTextureFormat::ETC2_RGBA,
            Self::Bc7 => basis_universal::TranscoderTextureFormat::BC7_RGBA,
            Self::Astc => basis_universal::TranscoderTextureFormat::ASTC_4x4_RGBA,
            Self::Rgba => basis_universal::TranscoderTextureFormat::RGBA32,
        }
    }

    fn as_wgpu(&self, srgb: bool) -> wgpu::TextureFormat {
        match self {
            Self::Etc2Rgba => {
                if srgb {
                    wgpu::TextureFormat::Etc2Rgba8UnormSrgb
                } else {
                    wgpu::TextureFormat::Etc2Rgba8Unorm
                }
            }
            Self::Astc => wgpu::TextureFormat::Astc {
                block: wgpu::AstcBlock::B4x4,
                channel: if srgb {
                    wgpu::AstcChannel::UnormSrgb
                } else {
                    wgpu::AstcChannel::Unorm
                },
            },
            Self::Rgba => {
                if srgb {
                    wgpu::TextureFormat::Rgba8UnormSrgb
                } else {
                    wgpu::TextureFormat::Rgba8Unorm
                }
            }
            Self::Bc7 => {
                if srgb {
                    wgpu::TextureFormat::Bc7RgbaUnormSrgb
                } else {
                    wgpu::TextureFormat::Bc7RgbaUnorm
                }
            }
        }
    }
}

fn load_basis(
    device: &Rc<wgpu::Device>,
    queue: &wgpu::Queue,
    supported_features: wgpu::Features,
    bytes: &[u8],
    srgb: bool,
) -> Texture {
    let array = unsafe { js_sys::Uint8Array::view(bytes) };

    let file = basis_universal_wasm::BasisFile::new(&array);

    let image = 0;
    let format = Format::new_from_features(supported_features);

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

async fn load_ktx2_async(
    context: &TextureLoadContext,
    srgb: bool,
    url: &Rc<url::Url>,
    binding: &MaterialTexture,
) -> anyhow::Result<Rc<Texture>> {
    // todo:
    // * At the moment it takes 3 round trips to load the first mip:
    //   The header, the level indices and then the first mip data.
    //
    //   This is very slow on high-latency connections. Instead of
    //   using range requests for these 3 pieces, we should instead
    //   request the whole file and use an abort controller
    //   to cancel it: https://javascript.info/fetch-abort.
    //
    // * We could also hide some of the latency while requesting the images
    //   while loading the large geometry blob.

    let mut header_bytes = [0; ktx2::Header::LENGTH];

    async_reader_from_fetch(url, Some(0..ktx2::Header::LENGTH))
        .await?
        .read_exact(&mut header_bytes)
        .await?;

    let header = ktx2::Header::from_bytes(&header_bytes);

    header.validate()?;

    let mut level_indices = Vec::with_capacity(header.level_count as usize);

    {
        let mut async_reader = async_reader_from_fetch(
            url,
            Some(
                ktx2::Header::LENGTH
                    ..ktx2::Header::LENGTH + ktx2::LevelIndex::LENGTH * header.level_count as usize,
            ),
        )
        .await?;

        for _ in 0..header.level_count {
            let mut level_index_bytes = [0; ktx2::LevelIndex::LENGTH];

            async_reader.read_exact(&mut level_index_bytes).await?;

            level_indices.push(ktx2::LevelIndex::from_bytes(&level_index_bytes));
        }
    }

    let transcoder = basis_universal::LowLevelUastcTranscoder::new();
    let format = Format::new_from_features(context.context.supported_features);

    let texture_descriptor = move || wgpu::TextureDescriptor {
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

    // Create a sampler that will only display mip levels from `level` onwards.
    let sampler_descriptor = move |level, context: &ModelLoadContext| wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        anisotropy_clamp: context.anisotropy_clamp,
        lod_min_clamp: level as f32,
        ..Default::default()
    };

    let texture = Rc::new(Texture::new(
        context.context.device.create_texture(&texture_descriptor()),
    ));

    let mut levels = level_indices.into_iter().enumerate().rev();

    // Load the smallest (1x1 pixel) mip first before returning the texture
    {
        let (i, level_index) = levels.next().unwrap();

        let url = Rc::clone(url);
        let textures = Rc::clone(&context.textures);
        let material_settings = Rc::clone(&context.material_settings);
        let bind_group = Rc::clone(&context.bind_group);
        let context = Rc::clone(&context.context);
        let texture = Rc::clone(&texture);
        let sampler = Rc::clone(&binding.sampler);

        let transcoded =
            decompress_and_transcode(&url, i as u32, &level_index, &header, &transcoder, format)
                .await
                .unwrap();

        write_bytes_to_texture(
            &context.queue,
            &texture.texture,
            i as u32,
            &transcoded,
            &texture_descriptor(),
        );

        *sampler.borrow_mut() = Rc::new(
            context
                .device
                .create_sampler(&sampler_descriptor(i, &context)),
        );

        let new_bind_group = create_model_bind_group(&context, &textures, &material_settings);

        *bind_group.borrow_mut() = new_bind_group;
    }

    // Load all other mips in the background.
    wasm_bindgen_futures::spawn_local({
        let url = Rc::clone(url);
        let textures = Rc::clone(&context.textures);
        let material_settings = Rc::clone(&context.material_settings);
        let bind_group = Rc::clone(&context.bind_group);
        let context = Rc::clone(&context.context);
        let texture = Rc::clone(&texture);
        let sampler = Rc::clone(&binding.sampler);

        async move {
            for (i, level_index) in levels {
                let transcoded = decompress_and_transcode(
                    &url,
                    i as u32,
                    &level_index,
                    &header,
                    &transcoder,
                    format,
                )
                .await
                .unwrap();

                write_bytes_to_texture(
                    &context.queue,
                    &texture.texture,
                    i as u32,
                    &transcoded,
                    &texture_descriptor(),
                );

                *sampler.borrow_mut() = Rc::new(
                    context
                        .device
                        .create_sampler(&sampler_descriptor(i, &context)),
                );

                let new_bind_group =
                    create_model_bind_group(&context, &textures, &material_settings);

                *bind_group.borrow_mut() = new_bind_group;
            }
        }
    });

    Ok(texture)
}

async fn decompress_and_transcode(
    url: &url::Url,
    level: u32,
    level_index: &ktx2::LevelIndex,
    header: &ktx2::Header,
    transcoder: &basis_universal::LowLevelUastcTranscoder,
    format: Format,
) -> anyhow::Result<Vec<u8>> {
    let mut async_read = async_reader_from_fetch(
        url,
        Some(level_index.offset as usize..(level_index.offset + level_index.length_bytes) as usize),
    )
    .await?;

    let decompressed = match header.supercompression_scheme {
        Some(ktx2::SupercompressionScheme::Zstandard) => {
            let level_reader = futures::io::BufReader::new(async_read);
            let mut decoder = async_compression::futures::bufread::ZstdDecoder::new(level_reader);
            read_num_bytes(&mut decoder, level_index.uncompressed_length_bytes as usize).await?
        }
        Some(other) => panic!("Unsupported: {:?}", other),
        None => {
            read_num_bytes(
                &mut async_read,
                level_index.uncompressed_length_bytes as usize,
            )
            .await?
        }
    };

    let slice_width = header.pixel_width >> level;
    let slice_height = header.pixel_height >> level;

    let (block_width_pixels, block_height_pixels) = (4, 4);

    let slice_parameters = basis_universal::SliceParametersUastc {
        num_blocks_x: ((slice_width + block_width_pixels - 1) / block_width_pixels).max(1),
        num_blocks_y: ((slice_height + block_height_pixels - 1) / block_height_pixels).max(1),
        has_alpha: false,
        original_width: slice_width,
        original_height: slice_height,
    };

    transcoder
        .transcode_slice(
            &decompressed,
            slice_parameters,
            basis_universal::DecodeFlags::HIGH_QUALITY,
            format.as_transcoder_block_format(),
        )
        .map_err(|err| anyhow::anyhow!("Transcoder error: {:?}", err))
}

async fn read_num_bytes<R: futures::io::AsyncRead + std::marker::Unpin>(
    reader: &mut R,
    num_bytes: usize,
) -> anyhow::Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(num_bytes);

    reader.read_to_end(&mut bytes).await?;

    Ok(bytes)
}

fn load_ktx2(
    device: &Rc<wgpu::Device>,
    queue: &wgpu::Queue,
    srgb: bool,
    supported_features: wgpu::Features,
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
    let format = Format::new_from_features(supported_features);

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

fn load_standard_image_format(
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
                                "../fullscreen_tri.spv"
                            ))
                        }),
                        entry_point: "fullscreen_tri",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: context.shader_cache.get("blit", || {
                            context
                                .device
                                .create_shader_module(&wgpu::include_spirv!("../blit.spv"))
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

pub(crate) fn load_single_pixel_image(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    bytes: &[u8; 4],
) -> Rc<Texture> {
    Rc::new(Texture::new(device.create_texture_with_data(
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
    )))
}

fn response_body_async_reader(
    response: web_sys::Response,
) -> anyhow::Result<impl futures::io::AsyncRead> {
    use futures::{StreamExt, TryStreamExt};

    let js_stream = wasm_streams::ReadableStream::from_raw(
        wasm_bindgen::JsValue::from(
            response
                .body()
                .ok_or_else(|| anyhow::anyhow!("Failed to get response body"))?,
        )
        .into(),
    );

    Ok(js_stream
        .into_stream()
        .map(|value| {
            let array: js_sys::Uint8Array = value.unwrap().into();
            let vec = array.to_vec();
            Ok(vec)
        })
        .into_async_read())
}

pub(crate) async fn async_reader_from_fetch(
    url: &url::Url,
    byte_range: Option<std::ops::Range<usize>>,
) -> anyhow::Result<impl futures::io::AsyncRead> {
    let mut request_init = web_sys::RequestInit::new();

    if let Some(byte_range) = byte_range {
        let headers = js_sys::Object::new();
        js_sys::Reflect::set(
            &headers,
            &"Range".into(),
            &format!("bytes={}-{}", byte_range.start, byte_range.end).into(),
        )
        .map_err(|err| anyhow::anyhow!("Js Error: {:?}", err))?;
        request_init.headers(&headers);
    }

    let response: web_sys::Response = wasm_bindgen_futures::JsFuture::from(
        web_sys::window()
            .unwrap()
            .fetch_with_str_and_init(url.as_str(), &request_init),
    )
    .await
    .map_err(|err| anyhow::anyhow!("{:?}", err))?
    .into();

    if !response.ok() {
        return Err(anyhow::anyhow!(
            "Bad fetch response:\nGot status code {} for {}",
            response.status(),
            url
        ));
    }

    let length = response
        .headers()
        .get("content-length")
        .map_err(|err| anyhow::anyhow!("{:?}", err))?
        .unwrap();

    let length: u64 = length.parse().unwrap();

    log::info!(
        "Fetching {}. Size in MB: {}",
        url,
        length as f32 / 1024.0 / 1024.0
    );

    response_body_async_reader(response)
}

async fn fetch_bytes(url: &url::Url) -> anyhow::Result<Vec<u8>> {
    let mut async_reader = async_reader_from_fetch(url, None).await?;

    let mut buf = Vec::new();

    async_reader.read_to_end(&mut buf).await?;

    Ok(buf)
}

pub(crate) type FetchedImages = std::collections::HashMap<Rc<url::Url>, (Rc<Texture>, bool)>;

pub(crate) fn prune_fetched_images(fetched_images: &mut FetchedImages) -> u32 {
    let mut removed = 0;

    fetched_images.retain(|_, (texture_ref, _)| {
        // Check the strong count. If a model is using the image then
        // it should be 2: the model + the one in this map. If less than 2
        // (it's normally impossible for strong_count to return 1 but w/e)
        // then we can drop it.
        if Rc::strong_count(texture_ref) < 2 {
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

// Like the following, except without trying to write subsequent mips.
// https://github.com/gfx-rs/wgpu/blob/0b61a191244da0f0d987d53614a6698097a7622f/wgpu/src/util/device.rs#L79-L146
fn create_texture_with_first_mip_data(
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

pub(crate) struct Texture {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
}

impl Texture {
    fn new(texture: wgpu::Texture) -> Self {
        Self {
            view: texture.create_view(&Default::default()),
            texture,
        }
    }
}
