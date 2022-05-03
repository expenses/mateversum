cd rust-gpu-cli-builder
cargo run --release -- ../shaders --multimodule
cd ..
for file in *.spv; do
    path=$(realpath $file)
    cd ../../spirv-extra-opt-passes/spirv-extra-opt
    #cargo run --quiet -- $path --experimental-remove-bad-op-switches -o $path;
    cd ../../vr/webxr-pbr
    spirv-val $file || echo $file;
done;