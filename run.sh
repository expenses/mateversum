cargo build --target wasm32-unknown-unknown --release &&
wasm-bindgen target/wasm32-unknown-unknown/release/webxr-pbr.wasm --out-dir web/pkg --target web &&
cd web &&
caddy file-server --listen :8000 --access-log