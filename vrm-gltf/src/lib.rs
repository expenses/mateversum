pub use gltf::json as gltf_json;

pub struct GltfExtensions;

impl gltf::json::CustomExtensions for GltfExtensions {
    type Root = RootExtensions;
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct RootExtensions {
    #[serde(default, rename = "VRM")]
    pub vrm: Option<Vrm>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct Vrm {
    #[serde(rename = "exporterVersion")]
    pub exporter_version: String,
    pub meta: Meta,
    pub humanoid: Humanoid,
    #[serde(rename = "firstPerson")]
    pub first_person: FirstPerson,
    #[serde(rename = "blendShapeMaster")]
    pub blend_shape_master: BlendShapeMaster,
    #[serde(rename = "secondaryAnimation")]
    pub secondary_animation: SecondaryAnimation,
    #[serde(rename = "materialProperties")]
    pub material_properties: Vec<MaterialProperty>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct Meta {
    pub version: String,
    pub author: String,
    #[serde(rename = "contactInformation")]
    pub contact_information: String,
    pub reference: String,
    pub title: String,
    pub texture: u32,
    #[serde(rename = "allowedUserName")]
    pub allowed_user_name: String,
    #[serde(rename = "violentUssageName")]
    pub violent_usage_name: String,
    // Oh dear
    #[serde(rename = "sexualUssageName")]
    pub sexual_usage_name: String,
    #[serde(rename = "commercialUssageName")]
    pub commercial_usage_name: String,
    #[serde(rename = "otherPermissionUrl")]
    pub other_permission_url: String,
    #[serde(rename = "licenseName")]
    pub license_name: String,
    #[serde(rename = "otherLicenseUrl")]
    pub other_license_url: String,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct Humanoid {
    #[serde(rename = "humanBones")]
    pub human_bones: Vec<Bone>,
    #[serde(rename = "armStretch")]
    pub arm_stretch: f32,
    #[serde(rename = "legStretch")]
    pub leg_stretch: f32,
    #[serde(rename = "upperArmTwist")]
    pub upper_arm_twist: f32,
    #[serde(rename = "lowerArmTwist")]
    pub lower_arm_twist: f32,
    #[serde(rename = "upperLegTwist")]
    pub upper_leg_twist: f32,
    #[serde(rename = "lowerLegTwist")]
    pub lower_leg_twist: f32,
    #[serde(rename = "feetSpacing")]
    pub feet_spacing: f32,
    #[serde(rename = "hasTranslationDoF")]
    pub has_translation_dof: bool,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct Bone {
    #[serde(rename = "bone")]
    pub name: String,
    pub node: u32,
    #[serde(rename = "useDefaultValues")]
    pub use_default_values: bool,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct FirstPerson {
    #[serde(rename = "firstPersonBone")]
    pub first_person_bone: u32,
    #[serde(rename = "firstPersonBoneOffset")]
    pub first_person_bone_offset: Vec3,
    // todo
    #[serde(rename = "meshAnnotations")]
    pub mesh_annotations: Vec<()>,
    #[serde(rename = "lookAtTypeName")]
    pub look_at_type_name: String,
    #[serde(rename = "lookAtHorizontalInner")]
    pub look_at_horizontal_inner: LookAtCurve,
    #[serde(rename = "lookAtHorizontalOuter")]
    pub look_at_horizontal_outer: LookAtCurve,
    #[serde(rename = "lookAtVerticalDown")]
    pub look_at_vertical_down: LookAtCurve,
    #[serde(rename = "lookAtVerticalUp")]
    pub look_at_vertical_up: LookAtCurve,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct LookAtCurve {
    pub curve: [u32; 8],
    #[serde(rename = "xRange")]
    pub x_range: u32,
    #[serde(rename = "yRange")]
    pub y_range: u32,
}

impl gltf::json::validation::Validate for LookAtCurve {}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct BlendShapeMaster {
    #[serde(rename = "blendShapeGroups")]
    pub blend_shape_groups: Vec<BlendShapeGroup>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct BlendShapeGroup {
    pub name: String,
    #[serde(rename = "presetName")]
    pub preset_name: String,
    pub binds: Vec<Bind>,
    // todo
    #[serde(rename = "materialValues")]
    pub material_values: Vec<()>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct Bind {
    pub mesh: u32,
    pub index: u32,
    pub weight: u32,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct SecondaryAnimation {
    #[serde(rename = "boneGroups")]
    pub bone_groups: Vec<BoneGroup>,
    #[serde(rename = "colliderGroups")]
    pub collider_groups: Vec<ColliderGroup>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct BoneGroup {
    pub comment: String,
    pub stiffiness: f32,
    #[serde(rename = "gravityPower")]
    pub gravity_power: f32,
    #[serde(rename = "gravityDir")]
    pub gravity_dir: Vec3,
    #[serde(rename = "dragForce")]
    pub drag_force: f32,
    pub center: f32,
    #[serde(rename = "hitRadius")]
    pub hit_radius: f32,
    pub bones: Vec<u32>,
    #[serde(rename = "colliderGroups")]
    pub collider_groups: Vec<u32>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct ColliderGroup {
    pub node: u32,
    pub colliders: Vec<Collider>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct Collider {
    pub offset: Vec3,
    pub radius: f32,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn as_array(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct MaterialProperty {
    pub name: String,
    #[serde(rename = "renderQueue")]
    pub render_queue: u32,
    pub shader: String,
    #[serde(rename = "floatProperties")]
    pub float: FloatProperties,
    #[serde(rename = "vectorProperties")]
    pub vector: VectorProperties,
    #[serde(rename = "textureProperties")]
    pub texture: TextureProperties,
    #[serde(rename = "keywordMap")]
    pub keyword_map: KeywordMap,
    #[serde(rename = "tagMap")]
    pub tag_map: TagMap,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct FloatProperties {
    #[serde(rename = "_ShadeShift")]
    pub shade_shift: f32,
    #[serde(rename = "_ShadeToony")]
    pub shade_toony: f32,
    #[serde(rename = "_Cutoff")]
    pub cutoff: f32,
    #[serde(rename = "_IndirectLightIntensity")]
    pub indirect_light_insensity: f32,
    #[serde(rename = "_OutlineWidth")]
    pub outline_width: f32,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct TextureProperties {
    #[serde(rename = "_MainTex")]
    pub main_tex: u32,
    #[serde(rename = "_ShadeTexture")]
    pub shade_texture: u32,
    #[serde(rename = "_BumpMap")]
    pub bump_map: u32,
    #[serde(rename = "_SphereAdd")]
    pub sphere_add: u32,
    #[serde(rename = "_EmissionMap")]
    pub emission_map: u32,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct VectorProperties {
    #[serde(rename = "_Color")]
    pub color: [f32; 4],
    #[serde(rename = "_ShadeColor")]
    pub shade_color: [f32; 4],
    #[serde(rename = "_OutlineColor")]
    pub outline_color: [f32; 4],
}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct TagMap {
    #[serde(rename = "RenderType")]
    pub render_type: RenderType,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum RenderType {
    Transparent,
    TransparentCutout,
    Opaque,
}

impl gltf::json::validation::Validate for RenderType {}

#[derive(serde::Deserialize, serde::Serialize, Debug, gltf::derive::Validate, Clone)]
pub struct KeywordMap {
    #[serde(rename = "_ALPHABLEND_ON")]
    pub alpha_blend: Option<bool>,
    #[serde(rename = "_ALPHATEST_ON")]
    pub alpha_test: Option<bool>,
    #[serde(rename = "_NORMALMAP")]
    pub normal_map: Option<bool>,
    #[serde(rename = "MTOON_OUTLINE_COLOR_FIXED")]
    pub outline_color_fixed: Option<bool>,
    #[serde(rename = "MTOON_OUTLINE_COLOR_MIXED")]
    pub outline_color_mixed: Option<bool>,
    #[serde(rename = "MTOON_OUTLINE_WIDTH_WORLD")]
    pub outline_width_world: Option<bool>,
}

#[test]
fn test() {
    let path = "../web/vrm-samples/vroid/Darkness_Shibu.vrm";

    let gltf = gltf::Gltf::<GltfExtensions>::from_slice(&std::fs::read(path).unwrap()).unwrap();

    let json = gltf.document.as_json();

    let extensions = json.extensions.as_ref().unwrap();

    let vrm = extensions.custom.vrm.as_ref().unwrap();
}
