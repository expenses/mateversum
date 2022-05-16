var __defProp = Object.defineProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};

// node_modules/trystero/src/index.js
var src_exports = {};
__export(src_exports, {
  joinRoom: () => joinRoom,
  selfId: () => selfId
});

// node_modules/simple-peer-light/index.js
var MAX_BUFFERED_AMOUNT = 64 * 1024;
var ICECOMPLETE_TIMEOUT = 5 * 1e3;
var CHANNEL_CLOSING_TIMEOUT = 5 * 1e3;
function randombytes(size) {
  const array = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    array[i] = Math.random() * 256 | 0;
  }
  return array;
}
function getBrowserRTC() {
  if (typeof globalThis === "undefined")
    return null;
  const wrtc = {
    RTCPeerConnection: globalThis.RTCPeerConnection || globalThis.mozRTCPeerConnection || globalThis.webkitRTCPeerConnection,
    RTCSessionDescription: globalThis.RTCSessionDescription || globalThis.mozRTCSessionDescription || globalThis.webkitRTCSessionDescription,
    RTCIceCandidate: globalThis.RTCIceCandidate || globalThis.mozRTCIceCandidate || globalThis.webkitRTCIceCandidate
  };
  if (!wrtc.RTCPeerConnection)
    return null;
  return wrtc;
}
function errCode(err, code) {
  Object.defineProperty(err, "code", {
    value: code,
    enumerable: true,
    configurable: true
  });
  return err;
}
function filterTrickle(sdp) {
  return sdp.replace(/a=ice-options:trickle\s\n/g, "");
}
function warn(message) {
  console.warn(message);
}
var Peer = class {
  constructor(opts = {}) {
    this._map = /* @__PURE__ */ new Map();
    this._id = randombytes(4).toString("hex").slice(0, 7);
    this._doDebug = opts.debug;
    this._debug("new peer %o", opts);
    this.channelName = opts.initiator ? opts.channelName || randombytes(20).toString("hex") : null;
    this.initiator = opts.initiator || false;
    this.channelConfig = opts.channelConfig || Peer.channelConfig;
    this.channelNegotiated = this.channelConfig.negotiated;
    this.config = Object.assign({}, Peer.config, opts.config);
    this.offerOptions = opts.offerOptions || {};
    this.answerOptions = opts.answerOptions || {};
    this.sdpTransform = opts.sdpTransform || ((sdp) => sdp);
    this.streams = opts.streams || (opts.stream ? [opts.stream] : []);
    this.trickle = opts.trickle !== void 0 ? opts.trickle : true;
    this.allowHalfTrickle = opts.allowHalfTrickle !== void 0 ? opts.allowHalfTrickle : false;
    this.iceCompleteTimeout = opts.iceCompleteTimeout || ICECOMPLETE_TIMEOUT;
    this.destroyed = false;
    this.destroying = false;
    this._connected = false;
    this.remoteAddress = void 0;
    this.remoteFamily = void 0;
    this.remotePort = void 0;
    this.localAddress = void 0;
    this.localFamily = void 0;
    this.localPort = void 0;
    this._wrtc = opts.wrtc && typeof opts.wrtc === "object" ? opts.wrtc : getBrowserRTC();
    if (!this._wrtc) {
      if (typeof window === "undefined") {
        throw errCode(new Error("No WebRTC support: Specify `opts.wrtc` option in this environment"), "ERR_WEBRTC_SUPPORT");
      } else {
        throw errCode(new Error("No WebRTC support: Not a supported browser"), "ERR_WEBRTC_SUPPORT");
      }
    }
    this._pcReady = false;
    this._channelReady = false;
    this._iceComplete = false;
    this._iceCompleteTimer = null;
    this._channel = null;
    this._pendingCandidates = [];
    this._isNegotiating = false;
    this._firstNegotiation = true;
    this._batchedNegotiation = false;
    this._queuedNegotiation = false;
    this._sendersAwaitingStable = [];
    this._senderMap = /* @__PURE__ */ new Map();
    this._closingInterval = null;
    this._remoteTracks = [];
    this._remoteStreams = [];
    this._chunk = null;
    this._cb = null;
    this._interval = null;
    try {
      this._pc = new this._wrtc.RTCPeerConnection(this.config);
    } catch (err) {
      this.destroy(errCode(err, "ERR_PC_CONSTRUCTOR"));
      return;
    }
    this._isReactNativeWebrtc = typeof this._pc._peerConnectionId === "number";
    this._pc.oniceconnectionstatechange = () => {
      this._onIceStateChange();
    };
    this._pc.onicegatheringstatechange = () => {
      this._onIceStateChange();
    };
    this._pc.onconnectionstatechange = () => {
      this._onConnectionStateChange();
    };
    this._pc.onsignalingstatechange = () => {
      this._onSignalingStateChange();
    };
    this._pc.onicecandidate = (event) => {
      this._onIceCandidate(event);
    };
    if (typeof this._pc.peerIdentity === "object") {
      this._pc.peerIdentity.catch((err) => {
        this.destroy(errCode(err, "ERR_PC_PEER_IDENTITY"));
      });
    }
    if (this.initiator || this.channelNegotiated) {
      this._setupData({
        channel: this._pc.createDataChannel(this.channelName, this.channelConfig)
      });
    } else {
      this._pc.ondatachannel = (event) => {
        this._setupData(event);
      };
    }
    if (this.streams) {
      this.streams.forEach((stream) => {
        this.addStream(stream);
      });
    }
    this._pc.ontrack = (event) => {
      this._onTrack(event);
    };
    this._debug("initial negotiation");
    this._needsNegotiation();
  }
  get bufferSize() {
    return this._channel && this._channel.bufferedAmount || 0;
  }
  get connected() {
    return this._connected && this._channel.readyState === "open";
  }
  address() {
    return {
      port: this.localPort,
      family: this.localFamily,
      address: this.localAddress
    };
  }
  signal(data) {
    if (this.destroying)
      return;
    if (this.destroyed)
      throw errCode(new Error("cannot signal after peer is destroyed"), "ERR_DESTROYED");
    if (typeof data === "string") {
      try {
        data = JSON.parse(data);
      } catch (err) {
        data = {};
      }
    }
    this._debug("signal()");
    if (data.renegotiate && this.initiator) {
      this._debug("got request to renegotiate");
      this._needsNegotiation();
    }
    if (data.transceiverRequest && this.initiator) {
      this._debug("got request for transceiver");
      this.addTransceiver(data.transceiverRequest.kind, data.transceiverRequest.init);
    }
    if (data.candidate) {
      if (this._pc.remoteDescription && this._pc.remoteDescription.type) {
        this._addIceCandidate(data.candidate);
      } else {
        this._pendingCandidates.push(data.candidate);
      }
    }
    if (data.sdp) {
      this._pc.setRemoteDescription(new this._wrtc.RTCSessionDescription(data)).then(() => {
        if (this.destroyed)
          return;
        this._pendingCandidates.forEach((candidate) => {
          this._addIceCandidate(candidate);
        });
        this._pendingCandidates = [];
        if (this._pc.remoteDescription.type === "offer")
          this._createAnswer();
      }).catch((err) => {
        this.destroy(errCode(err, "ERR_SET_REMOTE_DESCRIPTION"));
      });
    }
    if (!data.sdp && !data.candidate && !data.renegotiate && !data.transceiverRequest) {
      this.destroy(errCode(new Error("signal() called with invalid signal data"), "ERR_SIGNALING"));
    }
  }
  _addIceCandidate(candidate) {
    const iceCandidateObj = new this._wrtc.RTCIceCandidate(candidate);
    this._pc.addIceCandidate(iceCandidateObj).catch((err) => {
      if (!iceCandidateObj.address || iceCandidateObj.address.endsWith(".local")) {
        warn("Ignoring unsupported ICE candidate.");
      } else {
        this.destroy(errCode(err, "ERR_ADD_ICE_CANDIDATE"));
      }
    });
  }
  send(chunk) {
    if (this.destroying)
      return;
    if (this.destroyed)
      throw errCode(new Error("cannot send after peer is destroyed"), "ERR_DESTROYED");
    this._channel.send(chunk);
  }
  addTransceiver(kind, init) {
    if (this.destroying)
      return;
    if (this.destroyed)
      throw errCode(new Error("cannot addTransceiver after peer is destroyed"), "ERR_DESTROYED");
    this._debug("addTransceiver()");
    if (this.initiator) {
      try {
        this._pc.addTransceiver(kind, init);
        this._needsNegotiation();
      } catch (err) {
        this.destroy(errCode(err, "ERR_ADD_TRANSCEIVER"));
      }
    } else {
      this.emit("signal", {
        type: "transceiverRequest",
        transceiverRequest: { kind, init }
      });
    }
  }
  addStream(stream) {
    if (this.destroying)
      return;
    if (this.destroyed)
      throw errCode(new Error("cannot addStream after peer is destroyed"), "ERR_DESTROYED");
    this._debug("addStream()");
    stream.getTracks().forEach((track) => {
      this.addTrack(track, stream);
    });
  }
  addTrack(track, stream) {
    if (this.destroying)
      return;
    if (this.destroyed)
      throw errCode(new Error("cannot addTrack after peer is destroyed"), "ERR_DESTROYED");
    this._debug("addTrack()");
    const submap = this._senderMap.get(track) || /* @__PURE__ */ new Map();
    let sender = submap.get(stream);
    if (!sender) {
      sender = this._pc.addTrack(track, stream);
      submap.set(stream, sender);
      this._senderMap.set(track, submap);
      this._needsNegotiation();
    } else if (sender.removed) {
      throw errCode(new Error("Track has been removed. You should enable/disable tracks that you want to re-add."), "ERR_SENDER_REMOVED");
    } else {
      throw errCode(new Error("Track has already been added to that stream."), "ERR_SENDER_ALREADY_ADDED");
    }
  }
  replaceTrack(oldTrack, newTrack, stream) {
    if (this.destroying)
      return;
    if (this.destroyed)
      throw errCode(new Error("cannot replaceTrack after peer is destroyed"), "ERR_DESTROYED");
    this._debug("replaceTrack()");
    const submap = this._senderMap.get(oldTrack);
    const sender = submap ? submap.get(stream) : null;
    if (!sender) {
      throw errCode(new Error("Cannot replace track that was never added."), "ERR_TRACK_NOT_ADDED");
    }
    if (newTrack)
      this._senderMap.set(newTrack, submap);
    if (sender.replaceTrack != null) {
      sender.replaceTrack(newTrack);
    } else {
      this.destroy(errCode(new Error("replaceTrack is not supported in this browser"), "ERR_UNSUPPORTED_REPLACETRACK"));
    }
  }
  removeTrack(track, stream) {
    if (this.destroying)
      return;
    if (this.destroyed)
      throw errCode(new Error("cannot removeTrack after peer is destroyed"), "ERR_DESTROYED");
    this._debug("removeSender()");
    const submap = this._senderMap.get(track);
    const sender = submap ? submap.get(stream) : null;
    if (!sender) {
      throw errCode(new Error("Cannot remove track that was never added."), "ERR_TRACK_NOT_ADDED");
    }
    try {
      sender.removed = true;
      this._pc.removeTrack(sender);
    } catch (err) {
      if (err.name === "NS_ERROR_UNEXPECTED") {
        this._sendersAwaitingStable.push(sender);
      } else {
        this.destroy(errCode(err, "ERR_REMOVE_TRACK"));
      }
    }
    this._needsNegotiation();
  }
  removeStream(stream) {
    if (this.destroying)
      return;
    if (this.destroyed)
      throw errCode(new Error("cannot removeStream after peer is destroyed"), "ERR_DESTROYED");
    this._debug("removeSenders()");
    stream.getTracks().forEach((track) => {
      this.removeTrack(track, stream);
    });
  }
  _needsNegotiation() {
    this._debug("_needsNegotiation");
    if (this._batchedNegotiation)
      return;
    this._batchedNegotiation = true;
    queueMicrotask(() => {
      this._batchedNegotiation = false;
      if (this.initiator || !this._firstNegotiation) {
        this._debug("starting batched negotiation");
        this.negotiate();
      } else {
        this._debug("non-initiator initial negotiation request discarded");
      }
      this._firstNegotiation = false;
    });
  }
  negotiate() {
    if (this.destroying)
      return;
    if (this.destroyed)
      throw errCode(new Error("cannot negotiate after peer is destroyed"), "ERR_DESTROYED");
    if (this.initiator) {
      if (this._isNegotiating) {
        this._queuedNegotiation = true;
        this._debug("already negotiating, queueing");
      } else {
        this._debug("start negotiation");
        setTimeout(() => {
          this._createOffer();
        }, 0);
      }
    } else {
      if (this._isNegotiating) {
        this._queuedNegotiation = true;
        this._debug("already negotiating, queueing");
      } else {
        this._debug("requesting negotiation from initiator");
        this.emit("signal", {
          type: "renegotiate",
          renegotiate: true
        });
      }
    }
    this._isNegotiating = true;
  }
  destroy(err) {
    if (this.destroyed || this.destroying)
      return;
    this.destroying = true;
    this._debug("destroying (error: %s)", err && (err.message || err));
    queueMicrotask(() => {
      this.destroyed = true;
      this.destroying = false;
      this._debug("destroy (error: %s)", err && (err.message || err));
      this._connected = false;
      this._pcReady = false;
      this._channelReady = false;
      this._remoteTracks = null;
      this._remoteStreams = null;
      this._senderMap = null;
      clearInterval(this._closingInterval);
      this._closingInterval = null;
      clearInterval(this._interval);
      this._interval = null;
      this._chunk = null;
      this._cb = null;
      if (this._channel) {
        try {
          this._channel.close();
        } catch (err2) {
        }
        this._channel.onmessage = null;
        this._channel.onopen = null;
        this._channel.onclose = null;
        this._channel.onerror = null;
      }
      if (this._pc) {
        try {
          this._pc.close();
        } catch (err2) {
        }
        this._pc.oniceconnectionstatechange = null;
        this._pc.onicegatheringstatechange = null;
        this._pc.onsignalingstatechange = null;
        this._pc.onicecandidate = null;
        this._pc.ontrack = null;
        this._pc.ondatachannel = null;
      }
      this._pc = null;
      this._channel = null;
      if (err)
        this.emit("error", err);
      this.emit("close");
    });
  }
  _setupData(event) {
    if (!event.channel) {
      return this.destroy(errCode(new Error("Data channel event is missing `channel` property"), "ERR_DATA_CHANNEL"));
    }
    this._channel = event.channel;
    this._channel.binaryType = "arraybuffer";
    if (typeof this._channel.bufferedAmountLowThreshold === "number") {
      this._channel.bufferedAmountLowThreshold = MAX_BUFFERED_AMOUNT;
    }
    this.channelName = this._channel.label;
    this._channel.onmessage = (event2) => {
      this._onChannelMessage(event2);
    };
    this._channel.onbufferedamountlow = () => {
      this._onChannelBufferedAmountLow();
    };
    this._channel.onopen = () => {
      this._onChannelOpen();
    };
    this._channel.onclose = () => {
      this._onChannelClose();
    };
    this._channel.onerror = (err) => {
      this.destroy(errCode(err, "ERR_DATA_CHANNEL"));
    };
    let isClosing = false;
    this._closingInterval = setInterval(() => {
      if (this._channel && this._channel.readyState === "closing") {
        if (isClosing)
          this._onChannelClose();
        isClosing = true;
      } else {
        isClosing = false;
      }
    }, CHANNEL_CLOSING_TIMEOUT);
  }
  _startIceCompleteTimeout() {
    if (this.destroyed)
      return;
    if (this._iceCompleteTimer)
      return;
    this._debug("started iceComplete timeout");
    this._iceCompleteTimer = setTimeout(() => {
      if (!this._iceComplete) {
        this._iceComplete = true;
        this._debug("iceComplete timeout completed");
        this.emit("iceTimeout");
        this.emit("_iceComplete");
      }
    }, this.iceCompleteTimeout);
  }
  _createOffer() {
    if (this.destroyed)
      return;
    this._pc.createOffer(this.offerOptions).then((offer) => {
      if (this.destroyed)
        return;
      if (!this.trickle && !this.allowHalfTrickle) {
        offer.sdp = filterTrickle(offer.sdp);
      }
      offer.sdp = this.sdpTransform(offer.sdp);
      const sendOffer = () => {
        if (this.destroyed)
          return;
        const signal = this._pc.localDescription || offer;
        this._debug("signal");
        this.emit("signal", {
          type: signal.type,
          sdp: signal.sdp
        });
      };
      const onSuccess = () => {
        this._debug("createOffer success");
        if (this.destroyed)
          return;
        if (this.trickle || this._iceComplete)
          sendOffer();
        else
          this.once("_iceComplete", sendOffer);
      };
      const onError = (err) => {
        this.destroy(errCode(err, "ERR_SET_LOCAL_DESCRIPTION"));
      };
      this._pc.setLocalDescription(offer).then(onSuccess).catch(onError);
    }).catch((err) => {
      this.destroy(errCode(err, "ERR_CREATE_OFFER"));
    });
  }
  _requestMissingTransceivers() {
    if (this._pc.getTransceivers) {
      this._pc.getTransceivers().forEach((transceiver) => {
        if (!transceiver.mid && transceiver.sender.track && !transceiver.requested) {
          transceiver.requested = true;
          this.addTransceiver(transceiver.sender.track.kind);
        }
      });
    }
  }
  _createAnswer() {
    if (this.destroyed)
      return;
    this._pc.createAnswer(this.answerOptions).then((answer) => {
      if (this.destroyed)
        return;
      if (!this.trickle && !this.allowHalfTrickle) {
        answer.sdp = filterTrickle(answer.sdp);
      }
      answer.sdp = this.sdpTransform(answer.sdp);
      const sendAnswer = () => {
        if (this.destroyed)
          return;
        const signal = this._pc.localDescription || answer;
        this._debug("signal");
        this.emit("signal", {
          type: signal.type,
          sdp: signal.sdp
        });
        if (!this.initiator)
          this._requestMissingTransceivers();
      };
      const onSuccess = () => {
        if (this.destroyed)
          return;
        if (this.trickle || this._iceComplete)
          sendAnswer();
        else
          this.once("_iceComplete", sendAnswer);
      };
      const onError = (err) => {
        this.destroy(errCode(err, "ERR_SET_LOCAL_DESCRIPTION"));
      };
      this._pc.setLocalDescription(answer).then(onSuccess).catch(onError);
    }).catch((err) => {
      this.destroy(errCode(err, "ERR_CREATE_ANSWER"));
    });
  }
  _onConnectionStateChange() {
    if (this.destroyed)
      return;
    if (this._pc.connectionState === "failed") {
      this.destroy(errCode(new Error("Connection failed."), "ERR_CONNECTION_FAILURE"));
    }
  }
  _onIceStateChange() {
    if (this.destroyed)
      return;
    const iceConnectionState = this._pc.iceConnectionState;
    const iceGatheringState = this._pc.iceGatheringState;
    this._debug("iceStateChange (connection: %s) (gathering: %s)", iceConnectionState, iceGatheringState);
    this.emit("iceStateChange", iceConnectionState, iceGatheringState);
    if (iceConnectionState === "connected" || iceConnectionState === "completed") {
      this._pcReady = true;
      this._maybeReady();
    }
    if (iceConnectionState === "failed") {
      this.destroy(errCode(new Error("Ice connection failed."), "ERR_ICE_CONNECTION_FAILURE"));
    }
    if (iceConnectionState === "closed") {
      this.destroy(errCode(new Error("Ice connection closed."), "ERR_ICE_CONNECTION_CLOSED"));
    }
  }
  getStats(cb) {
    const flattenValues = (report) => {
      if (Object.prototype.toString.call(report.values) === "[object Array]") {
        report.values.forEach((value) => {
          Object.assign(report, value);
        });
      }
      return report;
    };
    if (this._pc.getStats.length === 0 || this._isReactNativeWebrtc) {
      this._pc.getStats().then((res) => {
        const reports = [];
        res.forEach((report) => {
          reports.push(flattenValues(report));
        });
        cb(null, reports);
      }, (err) => cb(err));
    } else if (this._pc.getStats.length > 0) {
      this._pc.getStats((res) => {
        if (this.destroyed)
          return;
        const reports = [];
        res.result().forEach((result) => {
          const report = {};
          result.names().forEach((name) => {
            report[name] = result.stat(name);
          });
          report.id = result.id;
          report.type = result.type;
          report.timestamp = result.timestamp;
          reports.push(flattenValues(report));
        });
        cb(null, reports);
      }, (err) => cb(err));
    } else {
      cb(null, []);
    }
  }
  _maybeReady() {
    this._debug("maybeReady pc %s channel %s", this._pcReady, this._channelReady);
    if (this._connected || this._connecting || !this._pcReady || !this._channelReady) {
      return;
    }
    this._connecting = true;
    const findCandidatePair = () => {
      if (this.destroyed)
        return;
      this.getStats((err, items) => {
        if (this.destroyed)
          return;
        if (err)
          items = [];
        const remoteCandidates = {};
        const localCandidates = {};
        const candidatePairs = {};
        let foundSelectedCandidatePair = false;
        items.forEach((item) => {
          if (item.type === "remotecandidate" || item.type === "remote-candidate") {
            remoteCandidates[item.id] = item;
          }
          if (item.type === "localcandidate" || item.type === "local-candidate") {
            localCandidates[item.id] = item;
          }
          if (item.type === "candidatepair" || item.type === "candidate-pair") {
            candidatePairs[item.id] = item;
          }
        });
        const setSelectedCandidatePair = (selectedCandidatePair) => {
          foundSelectedCandidatePair = true;
          let local = localCandidates[selectedCandidatePair.localCandidateId];
          if (local && (local.ip || local.address)) {
            this.localAddress = local.ip || local.address;
            this.localPort = Number(local.port);
          } else if (local && local.ipAddress) {
            this.localAddress = local.ipAddress;
            this.localPort = Number(local.portNumber);
          } else if (typeof selectedCandidatePair.googLocalAddress === "string") {
            local = selectedCandidatePair.googLocalAddress.split(":");
            this.localAddress = local[0];
            this.localPort = Number(local[1]);
          }
          if (this.localAddress) {
            this.localFamily = this.localAddress.includes(":") ? "IPv6" : "IPv4";
          }
          let remote = remoteCandidates[selectedCandidatePair.remoteCandidateId];
          if (remote && (remote.ip || remote.address)) {
            this.remoteAddress = remote.ip || remote.address;
            this.remotePort = Number(remote.port);
          } else if (remote && remote.ipAddress) {
            this.remoteAddress = remote.ipAddress;
            this.remotePort = Number(remote.portNumber);
          } else if (typeof selectedCandidatePair.googRemoteAddress === "string") {
            remote = selectedCandidatePair.googRemoteAddress.split(":");
            this.remoteAddress = remote[0];
            this.remotePort = Number(remote[1]);
          }
          if (this.remoteAddress) {
            this.remoteFamily = this.remoteAddress.includes(":") ? "IPv6" : "IPv4";
          }
          this._debug("connect local: %s:%s remote: %s:%s", this.localAddress, this.localPort, this.remoteAddress, this.remotePort);
        };
        items.forEach((item) => {
          if (item.type === "transport" && item.selectedCandidatePairId) {
            setSelectedCandidatePair(candidatePairs[item.selectedCandidatePairId]);
          }
          if (item.type === "googCandidatePair" && item.googActiveConnection === "true" || (item.type === "candidatepair" || item.type === "candidate-pair") && item.selected) {
            setSelectedCandidatePair(item);
          }
        });
        if (!foundSelectedCandidatePair && (!Object.keys(candidatePairs).length || Object.keys(localCandidates).length)) {
          setTimeout(findCandidatePair, 100);
          return;
        } else {
          this._connecting = false;
          this._connected = true;
        }
        if (this._chunk) {
          try {
            this.send(this._chunk);
          } catch (err2) {
            return this.destroy(errCode(err2, "ERR_DATA_CHANNEL"));
          }
          this._chunk = null;
          this._debug('sent chunk from "write before connect"');
          const cb = this._cb;
          this._cb = null;
          cb(null);
        }
        if (typeof this._channel.bufferedAmountLowThreshold !== "number") {
          this._interval = setInterval(() => this._onInterval(), 150);
          if (this._interval.unref)
            this._interval.unref();
        }
        this._debug("connect");
        this.emit("connect");
      });
    };
    findCandidatePair();
  }
  _onInterval() {
    if (!this._cb || !this._channel || this._channel.bufferedAmount > MAX_BUFFERED_AMOUNT) {
      return;
    }
    this._onChannelBufferedAmountLow();
  }
  _onSignalingStateChange() {
    if (this.destroyed)
      return;
    if (this._pc.signalingState === "stable") {
      this._isNegotiating = false;
      this._debug("flushing sender queue", this._sendersAwaitingStable);
      this._sendersAwaitingStable.forEach((sender) => {
        this._pc.removeTrack(sender);
        this._queuedNegotiation = true;
      });
      this._sendersAwaitingStable = [];
      if (this._queuedNegotiation) {
        this._debug("flushing negotiation queue");
        this._queuedNegotiation = false;
        this._needsNegotiation();
      } else {
        this._debug("negotiated");
        this.emit("negotiated");
      }
    }
    this._debug("signalingStateChange %s", this._pc.signalingState);
    this.emit("signalingStateChange", this._pc.signalingState);
  }
  _onIceCandidate(event) {
    if (this.destroyed)
      return;
    if (event.candidate && this.trickle) {
      this.emit("signal", {
        type: "candidate",
        candidate: {
          candidate: event.candidate.candidate,
          sdpMLineIndex: event.candidate.sdpMLineIndex,
          sdpMid: event.candidate.sdpMid
        }
      });
    } else if (!event.candidate && !this._iceComplete) {
      this._iceComplete = true;
      this.emit("_iceComplete");
    }
    if (event.candidate) {
      this._startIceCompleteTimeout();
    }
  }
  _onChannelMessage(event) {
    if (this.destroyed)
      return;
    let data = event.data;
    if (data instanceof ArrayBuffer)
      data = new Uint8Array(data);
    this.emit("data", data);
  }
  _onChannelBufferedAmountLow() {
    if (this.destroyed || !this._cb)
      return;
    this._debug("ending backpressure: bufferedAmount %d", this._channel.bufferedAmount);
    const cb = this._cb;
    this._cb = null;
    cb(null);
  }
  _onChannelOpen() {
    if (this._connected || this.destroyed)
      return;
    this._debug("on channel open");
    this._channelReady = true;
    this._maybeReady();
  }
  _onChannelClose() {
    if (this.destroyed)
      return;
    this._debug("on channel close");
    this.destroy();
  }
  _onTrack(event) {
    if (this.destroyed)
      return;
    event.streams.forEach((eventStream) => {
      this._debug("on track");
      this.emit("track", event.track, eventStream);
      this._remoteTracks.push({
        track: event.track,
        stream: eventStream
      });
      if (this._remoteStreams.some((remoteStream) => {
        return remoteStream.id === eventStream.id;
      })) {
        return;
      }
      this._remoteStreams.push(eventStream);
      queueMicrotask(() => {
        this._debug("on stream");
        this.emit("stream", eventStream);
      });
    });
  }
  _debug(...args) {
    if (!this._doDebug)
      return;
    args[0] = "[" + this._id + "] " + args[0];
    console.log(...args);
  }
  on(key, listener) {
    const map = this._map;
    if (!map.has(key))
      map.set(key, /* @__PURE__ */ new Set());
    map.get(key).add(listener);
  }
  off(key, listener) {
    const map = this._map;
    const listeners = map.get(key);
    if (!listeners)
      return;
    listeners.delete(listener);
    if (listeners.size === 0)
      map.delete(key);
  }
  once(key, listener) {
    const listener_ = (...args) => {
      this.off(key, listener_);
      listener(...args);
    };
    this.on(key, listener_);
  }
  emit(key, ...args) {
    const map = this._map;
    if (!map.has(key))
      return;
    for (const listener of map.get(key)) {
      try {
        listener(...args);
      } catch (err) {
        console.error(err);
      }
    }
  }
};
Peer.WEBRTC_SUPPORT = !!getBrowserRTC();
Peer.config = {
  iceServers: [
    {
      urls: [
        "stun:stun.l.google.com:19302",
        "stun:global.stun.twilio.com:3478"
      ]
    }
  ],
  sdpSemantics: "unified-plan"
};
Peer.channelConfig = {};
var simple_peer_light_default = Peer;

// node_modules/trystero/src/utils.js
var charSet = "0123456789AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz";
var initPeer = (initiator, trickle, config) => {
  const peer = new simple_peer_light_default({ initiator, trickle, config });
  const onData = (data) => peer.__earlyDataBuffer.push(data);
  peer.on(events.data, onData);
  peer.__earlyDataBuffer = [];
  peer.__drainEarlyData = (f) => {
    peer.off(events.data, onData);
    peer.__earlyDataBuffer.forEach(f);
    delete peer.__earlyDataBuffer;
    delete peer.__drainEarlyData;
  };
  return peer;
};
var genId = (n) => new Array(n).fill().map(() => charSet[Math.floor(Math.random() * charSet.length)]).join("");
var initGuard = (occupiedRooms2, f) => (config, ns) => {
  if (occupiedRooms2[ns]) {
    throw mkErr(`already joined room ${ns}`);
  }
  if (!config) {
    throw mkErr("requires a config map as the first argument");
  }
  if (!config.appId && !config.firebaseApp) {
    throw mkErr("config map is missing appId field");
  }
  if (!ns) {
    throw mkErr("namespace argument required");
  }
  return f(config, ns);
};
var libName = "Trystero";
var selfId = genId(20);
var { keys, values, entries, fromEntries } = Object;
var noOp = () => {
};
var mkErr = (msg) => new Error(`${libName}: ${msg}`);
var encodeBytes = (txt) => new TextEncoder().encode(txt);
var decodeBytes = (txt) => new TextDecoder().decode(txt);
var events = fromEntries(["close", "connect", "data", "error", "signal", "stream", "track"].map((k) => [
  k,
  k
]));
var combineChunks = (chunks) => {
  const full = new Uint8Array(chunks.reduce((a, c) => a + c.byteLength, 0));
  chunks.reduce((a, c) => {
    full.set(c, a);
    return a + c.byteLength;
  }, 0);
  return full;
};

// node_modules/trystero/src/room.js
var TypedArray = Object.getPrototypeOf(Uint8Array);
var typeByteLimit = 12;
var typeIndex = 0;
var nonceIndex = typeIndex + typeByteLimit;
var tagIndex = nonceIndex + 1;
var progressIndex = tagIndex + 1;
var payloadIndex = progressIndex + 1;
var chunkSize = 16 * 2 ** 10 - payloadIndex;
var oneByteMax = 255;
var buffLowEvent = "bufferedamountlow";
var room_default = (onPeer, onSelfLeave) => {
  const peerMap = {};
  const actions = {};
  const pendingTransmissions = {};
  const pendingPongs = {};
  const pendingStreamMetas = {};
  const pendingTrackMetas = {};
  const iterate = (targets, f) => (targets ? Array.isArray(targets) ? targets : [targets] : keys(peerMap)).flatMap((id) => {
    const peer = peerMap[id];
    if (!peer) {
      console.warn(`${libName}: no peer with id ${id} found`);
      return [];
    }
    return f(id, peer);
  });
  const exitPeer = (id) => {
    if (!peerMap[id]) {
      return;
    }
    delete peerMap[id];
    delete pendingTransmissions[id];
    delete pendingPongs[id];
    onPeerLeave(id);
  };
  const makeAction = (type) => {
    if (!type) {
      throw mkErr("action type argument is required");
    }
    const typeEncoded = encodeBytes(type);
    if (typeEncoded.byteLength > typeByteLimit) {
      throw mkErr(`action type string "${type}" (${typeEncoded.byteLength}b) exceeds byte limit (${typeByteLimit}). Hint: choose a shorter name.`);
    }
    const typeBytes = new Uint8Array(typeByteLimit);
    typeBytes.set(typeEncoded);
    const typePadded = decodeBytes(typeBytes);
    if (actions[typePadded]) {
      throw mkErr(`action '${type}' already registered`);
    }
    let nonce = 0;
    actions[typePadded] = { onComplete: noOp, onProgress: noOp };
    return [
      async (data, targets, meta, onProgress) => {
        if (meta && typeof meta !== "object") {
          throw mkErr("action meta argument must be an object");
        }
        if (data === void 0) {
          throw mkErr("action data cannot be undefined");
        }
        const isJson = typeof data !== "string";
        const isBlob = data instanceof Blob;
        const isBinary = isBlob || data instanceof ArrayBuffer || data instanceof TypedArray;
        if (meta && !isBinary) {
          throw mkErr("action meta argument can only be used with binary data");
        }
        const buffer = isBinary ? new Uint8Array(isBlob ? await data.arrayBuffer() : data) : encodeBytes(isJson ? JSON.stringify(data) : data);
        const metaEncoded = meta ? encodeBytes(JSON.stringify(meta)) : null;
        const chunkTotal = Math.ceil(buffer.byteLength / chunkSize) + (meta ? 1 : 0);
        const chunks = new Array(chunkTotal).fill().map((_, i) => {
          const isLast = i === chunkTotal - 1;
          const isMeta = meta && i === 0;
          const chunk = new Uint8Array(payloadIndex + (isMeta ? metaEncoded.byteLength : isLast ? buffer.byteLength - chunkSize * (chunkTotal - (meta ? 2 : 1)) : chunkSize));
          chunk.set(typeBytes);
          chunk.set([nonce], nonceIndex);
          chunk.set([isLast | isMeta << 1 | isBinary << 2 | isJson << 3], tagIndex);
          chunk.set([Math.round((i + 1) / chunkTotal * oneByteMax)], progressIndex);
          chunk.set(meta ? isMeta ? metaEncoded : buffer.subarray((i - 1) * chunkSize, i * chunkSize) : buffer.subarray(i * chunkSize, (i + 1) * chunkSize), payloadIndex);
          return chunk;
        });
        nonce = nonce + 1 & oneByteMax;
        return Promise.all(iterate(targets, async (id, peer) => {
          const chan = peer._channel;
          let chunkN = 0;
          while (chunkN < chunkTotal) {
            const chunk = chunks[chunkN];
            if (chan.bufferedAmount > chan.bufferedAmountLowThreshold) {
              await new Promise((res) => {
                const next = () => {
                  chan.removeEventListener(buffLowEvent, next);
                  res();
                };
                chan.addEventListener(buffLowEvent, next);
              });
            }
            if (!peerMap[id]) {
              break;
            }
            peer.send(chunk);
            chunkN++;
            if (onProgress) {
              onProgress(chunk[progressIndex] / oneByteMax, id, meta);
            }
          }
        }));
      },
      (onComplete) => actions[typePadded] = { ...actions[typePadded], onComplete },
      (onProgress) => actions[typePadded] = { ...actions[typePadded], onProgress }
    ];
  };
  const handleData = (id, data) => {
    const buffer = new Uint8Array(data);
    const type = decodeBytes(buffer.subarray(typeIndex, nonceIndex));
    const [nonce] = buffer.subarray(nonceIndex, tagIndex);
    const [tag] = buffer.subarray(tagIndex, progressIndex);
    const [progress] = buffer.subarray(progressIndex, payloadIndex);
    const payload = buffer.subarray(payloadIndex);
    const isLast = !!(tag & 1);
    const isMeta = !!(tag & 1 << 1);
    const isBinary = !!(tag & 1 << 2);
    const isJson = !!(tag & 1 << 3);
    if (!actions[type]) {
      throw mkErr(`received message with unregistered type (${type})`);
    }
    if (!pendingTransmissions[id]) {
      pendingTransmissions[id] = {};
    }
    if (!pendingTransmissions[id][type]) {
      pendingTransmissions[id][type] = {};
    }
    let target = pendingTransmissions[id][type][nonce];
    if (!target) {
      target = pendingTransmissions[id][type][nonce] = { chunks: [] };
    }
    if (isMeta) {
      target.meta = JSON.parse(decodeBytes(payload));
    } else {
      target.chunks.push(payload);
    }
    actions[type].onProgress(progress / oneByteMax, id, target.meta);
    if (!isLast) {
      return;
    }
    const full = combineChunks(target.chunks);
    if (isBinary) {
      actions[type].onComplete(full, id, target.meta);
    } else {
      const text = decodeBytes(full);
      actions[type].onComplete(isJson ? JSON.parse(text) : text, id);
    }
    delete pendingTransmissions[id][type][nonce];
  };
  const [sendPing, getPing] = makeAction("__91n6__");
  const [sendPong, getPong] = makeAction("__90n6__");
  const [sendSignal, getSignal] = makeAction("__516n4L__");
  const [sendStreamMeta, getStreamMeta] = makeAction("__57r34m__");
  const [sendTrackMeta, getTrackMeta] = makeAction("__7r4ck__");
  let onPeerJoin = noOp;
  let onPeerLeave = noOp;
  let onPeerStream = noOp;
  let onPeerTrack = noOp;
  onPeer((peer, id) => {
    if (peerMap[id]) {
      return;
    }
    const onData = handleData.bind(null, id);
    peerMap[id] = peer;
    peer.on(events.signal, (sdp) => sendSignal(sdp, id));
    peer.on(events.close, () => exitPeer(id));
    peer.on(events.data, onData);
    peer.on(events.stream, (stream) => {
      onPeerStream(stream, id, pendingStreamMetas[id]);
      delete pendingStreamMetas[id];
    });
    peer.on(events.track, (track, stream) => {
      onPeerTrack(track, stream, id, pendingTrackMetas[id]);
      delete pendingTrackMetas[id];
    });
    peer.on(events.error, (e) => {
      if (e.code === "ERR_DATA_CHANNEL") {
        return;
      }
      console.error(e);
    });
    onPeerJoin(peer, id);
    peer.__drainEarlyData(onData);
  });
  getPing((_, id) => sendPong(null, id));
  getPong((_, id) => {
    if (pendingPongs[id]) {
      pendingPongs[id]();
      delete pendingPongs[id];
    }
  });
  getSignal((sdp, id) => {
    if (peerMap[id]) {
      peerMap[id].signal(sdp);
    }
  });
  getStreamMeta((meta, id) => pendingStreamMetas[id] = meta);
  getTrackMeta((meta, id) => pendingTrackMetas[id] = meta);
  return {
    makeAction,
    ping: async (id) => {
      if (!id) {
        throw mkErr("ping() must be called with target peer ID");
      }
      const start = Date.now();
      sendPing(null, id);
      await new Promise((res) => pendingPongs[id] = res);
      return Date.now() - start;
    },
    leave: () => {
      entries(peerMap).forEach(([id, peer]) => {
        peer.destroy();
        delete peerMap[id];
      });
      onSelfLeave();
    },
    getPeers: () => keys(peerMap),
    addStream: (stream, targets, meta) => iterate(targets, async (id, peer) => {
      if (meta) {
        await sendStreamMeta(meta, id);
      }
      peer.addStream(stream);
    }),
    removeStream: (stream, targets) => iterate(targets, (_, peer) => peer.removeStream(stream)),
    addTrack: (track, stream, targets, meta) => iterate(targets, async (id, peer) => {
      if (meta) {
        await sendTrackMeta(meta, id);
      }
      peer.addTrack(track, stream);
    }),
    removeTrack: (track, stream, targets) => iterate(targets, (_, peer) => peer.removeTrack(track, stream)),
    replaceTrack: (oldTrack, newTrack, stream, targets, meta) => iterate(targets, async (id, peer) => {
      if (meta) {
        await sendTrackMeta(meta, id);
      }
      peer.replaceTrack(oldTrack, newTrack, stream);
    }),
    onPeerJoin: (f) => onPeerJoin = f,
    onPeerLeave: (f) => onPeerLeave = f,
    onPeerStream: (f) => onPeerStream = f,
    onPeerTrack: (f) => onPeerTrack = f
  };
};

// node_modules/trystero/src/crypto.js
var algo = "AES-CBC";
var ecdsa_params = {
  name: "ECDSA",
  hash: { name: "SHA-384" }
};
var ecdsa_import_params = {
  name: "ECDSA",
  namedCurve: "P-384"
};
var pack = (buff) => window.btoa(String.fromCharCode.apply(null, new Uint8Array(buff)));
var unpack = (packed) => {
  const str = window.atob(packed);
  return new Uint8Array(str.length).map((_, i) => str.charCodeAt(i)).buffer;
};
var genKey = async (secret, ns) => crypto.subtle.importKey("raw", await crypto.subtle.digest({ name: "SHA-256" }, encodeBytes(`${secret}:${ns}`)), { name: algo }, false, ["encrypt", "decrypt"]);
var sign = async (key_pair, sdp) => {
  const encoder = new TextEncoder();
  const encoded_sdp = encoder.encode(sdp);
  const signature = await crypto.subtle.sign(ecdsa_params, key_pair.privateKey, encoded_sdp);
  const exported_key = await crypto.subtle.exportKey("jwk", key_pair.publicKey);
  return JSON.stringify({
    sdp,
    signature: pack(signature),
    key: exported_key
  });
};
var verify = async (string) => {
  const data = JSON.parse(string);
  const imported_key = await crypto.subtle.importKey("jwk", data.key, ecdsa_import_params, true, ["verify"]);
  const encoder = new TextEncoder();
  const encoded_sdp = encoder.encode(data.sdp);
  const signature = unpack(data.signature);
  const verified = await window.crypto.subtle.verify(ecdsa_params, imported_key, signature, encoded_sdp);
  return {
    sdp: data.sdp,
    verified,
    key: imported_key
  };
};

// node_modules/trystero/src/torrent.js
var occupiedRooms = {};
var sockets = {};
var socketListeners = {};
var hashLimit = 20;
var offerPoolSize = 10;
var defaultRedundancy = 2;
var defaultAnnounceSecs = 33;
var maxAnnounceSecs = 120;
var trackerAction = "announce";
var defaultTrackerUrls = [
  "wss://tracker.openwebtorrent.com",
  "wss://tracker.btorrent.xyz",
  "wss://tracker.files.fm:7073/announce",
  "wss://spacetradersapi-chatbox.herokuapp.com:443/announce"
];
var joinRoom = initGuard(occupiedRooms, (config, ns) => {
  const connectedPeers = {};
  const key = config.password && genKey(config.password, ns);
  const trackerUrls = (config.trackerUrls || defaultTrackerUrls).slice(0, config.trackerUrls ? config.trackerUrls.length : config.trackerRedundancy || defaultRedundancy);
  if (!trackerUrls.length) {
    throw mkErr("trackerUrls is empty");
  }
  const infoHashP = crypto.subtle.digest("SHA-1", encodeBytes(`${libName}:${config.appId}:${ns}`)).then((buffer) => Array.from(new Uint8Array(buffer)).map((b) => b.toString(36)).join("").slice(0, hashLimit));
  const makeOffers = () => fromEntries(new Array(offerPoolSize).fill().map(() => {
    const peer = initPeer(true, false, config.rtcConfig);
    return [
      genId(hashLimit),
      { peer, offerP: new Promise((res) => peer.once(events.signal, res)) }
    ];
  }));
  const onSocketMessage = async (socket, e) => {
    const infoHash = await infoHashP;
    let val;
    try {
      val = JSON.parse(e.data);
    } catch (e2) {
      console.error(`${libName}: received malformed SDP JSON`);
      return;
    }
    if (val.info_hash !== infoHash || val.peer_id && val.peer_id === selfId) {
      return;
    }
    const failure = val["failure reason"];
    if (failure) {
      console.warn(`${libName}: torrent tracker failure (${failure})`);
      return;
    }
    if (val.interval && val.interval > announceSecs && val.interval <= maxAnnounceSecs) {
      clearInterval(announceInterval);
      announceSecs = val.interval;
      announceInterval = setInterval(announceAll, announceSecs * 1e3);
    }
    if (val.offer && val.offer_id) {
      if (connectedPeers[val.peer_id] || handledOffers[val.offer_id]) {
        return;
      }
      handledOffers[val.offer_id] = true;
      const peer = initPeer(false, false, config.rtcConfig);
      peer.once(events.signal, async (answer) => socket.send(JSON.stringify({
        answer: { ...answer, sdp: await sign(config.signing_key, answer.sdp) },
        action: trackerAction,
        info_hash: infoHash,
        peer_id: selfId,
        to_peer_id: val.peer_id,
        offer_id: val.offer_id
      })));
      peer.on(events.connect, () => onConnect(peer, val.peer_id));
      peer.on(events.close, () => onDisconnect(val.peer_id));
      const verification_result = await verify(val.offer.sdp);
      if (verification_result.verified) {
        peer.signal({ ...val.offer, sdp: verification_result.sdp });
        console.log(`Verified ${val.peer_id}`);
        peer.key = verification_result.key;
      } else {
        console.warn(`Peer ${val.peer_id} sent a SDP that failed verification.`);
      }
      return;
    }
    if (val.answer) {
      if (connectedPeers[val.peer_id] || handledOffers[val.offer_id]) {
        return;
      }
      const offer = offerPool[val.offer_id];
      if (offer) {
        const { peer } = offer;
        if (peer.destroyed) {
          return;
        }
        handledOffers[val.offer_id] = true;
        peer.on(events.connect, () => onConnect(peer, val.peer_id, val.offer_id));
        peer.on(events.close, () => onDisconnect(val.peer_id));
        const verification_result = await verify(val.answer.sdp);
        if (verification_result.verified) {
          peer.signal({ ...val.answer, sdp: verification_result.sdp });
          console.log(`Verified ${val.peer_id}`);
          peer.key = verification_result.key;
        } else {
          console.warn(`Peer ${val.peer_id} sent a SDP that failed verification.`);
        }
      }
    }
  };
  const announce = async (socket, infoHash) => socket.send(JSON.stringify({
    action: trackerAction,
    info_hash: infoHash,
    numwant: offerPoolSize,
    peer_id: selfId,
    offers: await Promise.all(entries(offerPool).map(async ([id, { offerP }]) => {
      const offer = await offerP;
      return {
        offer_id: id,
        offer: { ...offer, sdp: await sign(config.signing_key, offer.sdp) }
      };
    }))
  }));
  const makeSocket = (url, infoHash, forced) => {
    if (forced || !sockets[url]) {
      socketListeners[url] = {
        ...socketListeners[url],
        [infoHash]: onSocketMessage
      };
      sockets[url] = new Promise((res) => {
        const socket = new WebSocket(url);
        socket.onopen = res.bind(null, socket);
        socket.onmessage = (e) => values(socketListeners[url]).forEach((f) => f(socket, e));
      });
    } else {
      socketListeners[url][infoHash] = onSocketMessage;
    }
    return sockets[url];
  };
  const announceAll = async () => {
    const infoHash = await infoHashP;
    if (offerPool) {
      cleanPool();
    }
    offerPool = makeOffers();
    trackerUrls.forEach(async (url) => {
      const socket = await makeSocket(url, infoHash);
      if (socket.readyState === WebSocket.OPEN) {
        announce(socket, infoHash);
      } else if (socket.readyState !== WebSocket.CONNECTING) {
        announce(await makeSocket(url, infoHash, true), infoHash);
      }
    });
  };
  const cleanPool = () => {
    entries(offerPool).forEach(([id, { peer }]) => {
      if (!handledOffers[id] && !connectedPeers[id]) {
        peer.destroy();
      }
    });
    handledOffers = {};
  };
  const onConnect = (peer, id, offerId) => {
    onPeerConnect(peer, id);
    connectedPeers[id] = true;
    if (offerId) {
      connectedPeers[offerId] = true;
    }
  };
  const onDisconnect = (id) => delete connectedPeers[id];
  let announceSecs = defaultAnnounceSecs;
  let announceInterval = setInterval(announceAll, announceSecs * 1e3);
  let onPeerConnect = noOp;
  let handledOffers = {};
  let offerPool;
  occupiedRooms[ns] = true;
  announceAll();
  return room_default((f) => onPeerConnect = f, async () => {
    const infoHash = await infoHashP;
    trackerUrls.forEach((url) => delete socketListeners[url][infoHash]);
    delete occupiedRooms[ns];
    clearInterval(announceInterval);
    cleanPool();
  });
});
export {
  src_exports as trystero
};
/*! simple-peer. MIT License. Feross Aboukhadijeh <https://feross.org/opensource> */
