pub struct ResourceCache<T> {
    inner: elsa::FrozenMap<&'static str, Box<T>>,
}

impl<T> Default for ResourceCache<T> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<T> ResourceCache<T> {
    pub fn get<F: Fn() -> T>(&self, key: &'static str, func: F) -> &T {
        if let Some(resource) = self.inner.get(key) {
            resource
        } else {
            log::info!("Generating {}", key);

            self.inner.insert(key, Box::new(func()))
        }
    }
}

pub struct PipelineData {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
}
