rm compiled-shaders/*

cd rust-gpu-cli-builder
cargo run --release -- ../shaders --multimodule --output ../compiled-shaders --capabilities MultiView --extensions SPV_KHR_multiview
cd ..
for file in compiled-shaders/*.spv; do
    path=$(realpath $file)
    cd ../../spirv-extra-opt-passes/spirv-extra-opt
    cargo run --quiet -- $path --experimental-remove-bad-op-switches -o $path;
    cd ../../vr/webxr-pbr
    spirv-val $file || echo $file;
done;

glslc granite-shaders/bc6.frag -o compiled-shaders/bc6.spv
spirv-opt compiled-shaders/bc6.spv -O -o compiled-shaders/bc6.spv

spirv-location-injector compiled-shaders/vertex.spv compiled-shaders/fragment_unlit.spv compiled-shaders/fragment_unlit.spv
spirv-location-injector compiled-shaders/vertex.spv compiled-shaders/fragment_unlit_alpha_clipped.spv compiled-shaders/fragment_unlit_alpha_clipped.spv
spirv-location-injector compiled-shaders/vertex.spv compiled-shaders/fragment_ui.spv compiled-shaders/fragment_ui.spv
spirv-location-injector compiled-shaders/vertex.spv compiled-shaders/flat_blue.spv compiled-shaders/flat_blue.spv
spirv-location-injector compiled-shaders/fullscreen_tri.spv compiled-shaders/bc6.spv compiled-shaders/bc6.spv
