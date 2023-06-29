import React, { useEffect, useRef, useState } from 'react'

const SwappedCamera = () => {
  const releaseCamera = async () => {
    console.log('????????')
    await window.api.releaseCamera()
  }
  useEffect(() => {
    return () => {
      releaseCamera()
    }
  }, [])
  return <img width="50" height="50" src="http://localhost:8000/swapped_camera" />
}

export default SwappedCamera
