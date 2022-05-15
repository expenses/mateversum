# Mateversum 🍾

Mateversum (pronounced: MAH-tay-ver-sum) is a peer-to-peer WebXR metaverse project.

The idea is that you'd be able to connect to a network of peers, load into a room, choose an avatar and hang out, all without any central servers. Inside a web browser.

Some of this sort-of works at the moment, but it's got a long way to go.

![](readme/networking.png)

_Real time networking between a browser on a laptop running the [WebXR API Emulator](https://chrome.google.com/webstore/detail/webxr-api-emulator/mjddjgeghkdijejnciaefnkjmkafnnje) extension and a browser on an android phone running in AR mode. I threw this very quickly so the lighting and bottle model could be better._

## A Few Implementation Details

We support models in the brilliant [glTF](https://github.com/KhronosGroup/glTF) format. The core of glTF is a JSON file that references both binary geometry data blobs and texture files as external URLs. These URLs can be delivered either by standard HTTP/HTTPS or ([IPFS](https://ipfs.io/) (currently only through a gateway but we'll hopefully use [js.ipfs.io](https://js.ipfs.io/) soon.)). We support textures files in the [KTX 2](https://github.khronos.org/KTX-Specification/) format. This lets us load textures progressively up from 1x1 pixel [mipmaps](https://en.wikipedia.org/wiki/Mipmap) all the way to 4096x4096 and beyond (with adjustable limits so that you can control bandwidth and memory).

Peer-to-peer networking is done via WebRTC. It's extremely bare-bones at the moment, with everyone just connecting to a single room and sending the position of their hand and hands to everyone else. WebRTC requires you to do a handshake where one peer sends a session descriptor offer packet to another peer, who then sends an answer packet back.

The transmission of these packets has to happen over an existing channel, usually using what's known as a signalling server. We're currently using Webtorrent trackers as signalling servers using the [trystero](https://github.com/dmotz/trystero) library, but using a [custom fork](https://github.com/expenses/trystero/tree/key-signing) that cryptographically signs the packets to prevent a bad server from performing a Man-in-the-middle attack (MITM).

Ideally we'd use libp2p for most if not all networking, but their WebRTC support isn't especially stable yet. See https://github.com/libp2p/specs/issues/220.