import React, { useEffect, useRef, useState } from 'react'

const CAPTURE_OPTIONS = {
  audio: false,
  video: true
}

const Camera = () => {
  const [mediaStream, setMediaStream] = useState(null)
  const videoRef = useRef()
  useEffect(() => {
    async function enableVideoStream() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(CAPTURE_OPTIONS)
        setMediaStream(stream)
        // Create a peer connection to send the stream to Python
        const pc = new RTCPeerConnection()
        console.log('pc', pc)
        mediaStream.getTracks().forEach((track) => {
          pc.addTrack(track, mediaStream)
        })
        pc.createOffer()
          .then((offer) => {
            console.log('??????', offer)
            return pc.setLocalDescription(offer)
          })
          .then(() => {
            const offer = pc.localDescription
            // TODO: Send the offer to Python using a WebSocket or HTTP request
            console.log('offer', offer)
          })
          .catch((error) => {
            console.log('??????SSSS')
            console.error('createOffer error:', error)
          })
      } catch (err) {
        // Handle the error
      }
    }

    if (!mediaStream) {
      enableVideoStream()
    } else {
      return function cleanup() {
        mediaStream.getTracks().forEach((track) => {
          track.stop()
        })
      }
    }
  }, [mediaStream])
  if (mediaStream && videoRef.current && !videoRef.current.srcObject) {
    videoRef.current.srcObject = mediaStream
  }

  return <video width="100%" height="100%" ref={videoRef} autoPlay playsInline muted />
}

export default Camera
