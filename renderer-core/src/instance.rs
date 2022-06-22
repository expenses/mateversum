use glam::Vec3;

#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Instance {
    pub position: Vec3,
    pub scale: f32,
    pub rotation: glam::Quat,
}

impl Instance {
    pub fn new(position: Vec3, scale: f32, rotation: glam::Quat) -> Self {
        Self {
            position,
            scale,
            rotation,
        }
    }

    pub fn from_transform(transform: web_sys::XrRigidTransform, scale: f32) -> Self {
        let rotation = transform.orientation();

        let rotation =
            glam::DQuat::from_xyzw(rotation.x(), rotation.y(), rotation.z(), rotation.w());
        Self {
            position: transform_to_position_vec3(&transform),
            rotation: rotation.as_f32(),
            scale,
        }
    }
}

impl Default for Instance {
    fn default() -> Self {
        Self::new(Vec3::ZERO, 1.0, glam::Quat::IDENTITY)
    }
}

fn transform_to_position_vec3(transform: &web_sys::XrRigidTransform) -> Vec3 {
    let position = transform.position();
    let position = glam::DVec3::new(position.x(), position.y(), position.z());
    position.as_vec3()
}
