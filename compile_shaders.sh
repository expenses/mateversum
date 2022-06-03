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
