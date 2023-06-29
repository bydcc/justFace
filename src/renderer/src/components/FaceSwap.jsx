import {
  Button,
  Checkbox,
  Col,
  Form,
  Input,
  Layout,
  Menu,
  Progress,
  Radio,
  Row,
  Space,
  theme
} from 'antd'
import React, { useEffect, useState } from 'react'
import { imageSuffix, videoSuffix } from '../../../const'

import Camera from './Camera'
import ReactPlayer from 'react-player'
import Store from '../Store'
import SwappedCamera from './SwappedCamera'
import { useSnapshot } from 'valtio'

const { Header, Footer, Sider, Content } = Layout

const FaceSwap = () => {
  const snap = useSnapshot(Store)
  window.api.handleUpdateProgress((event, value) => {
    Store.processProgress = value
  })
  const chooseSourceFile = async () => {
    let filePaths = await window.api.chooseFileOrFolder(
      ['openFile'],
      [
        {
          name: 'media',
          extensions: videoSuffix.concat(imageSuffix)
        }
      ]
    )
    Store.sourceFilePath = filePaths[0]
    let sourceFileSuffix = filePaths[0].substring(filePaths[0].lastIndexOf('.') + 1)
    console.log('sourceFileSuffix', sourceFileSuffix, videoSuffix)
    if (videoSuffix.includes(sourceFileSuffix)) {
      Store.sourceFileType = 'video'
    } else {
      Store.sourceFileType = 'image'
    }
  }

  const chooseTargetFile = async () => {
    let filePaths = await window.api.chooseFileOrFolder(
      ['openFile'],
      [
        {
          name: 'media',
          extensions: imageSuffix
        }
      ]
    )
    Store.targetFilePath = filePaths[0]
  }

  const startProcess = () => {
    if (snap.sourceType === 'multimedia') {
      window.api.startProcess({
        sourceFilePath: snap.sourceFilePath,
        sourceFileType: snap.sourceFileType,
        targetFilePath: snap.targetFilePath,
        modelFolderPath: snap.modelFolderPath
      })
    }
  }

  const cancelProcess = () => {
    window.api.cancelProcess()
    Store.processProgress = 0
    Store.processing = false
  }

  const selectSourceType = (e) => {
    const value = e.target.value
    Store.sourceType = value
    if (value !== 'camera') {
      window.api.releaseCamera()
    }
  }

  const onChangeConfig = (checkedValues) => {
    console.log('checked = ', checkedValues)
  }

  useEffect(() => {
    // Update the document title using the browser API
  })

  return (
    <div
      style={{
        height: '100%'
      }}
    >
      <Row
        style={{
          height: '50%'
        }}
        gutter={[8, 8]}
      >
        <Col span={12}>
          <Layout
            style={{
              display: 'flex',
              alignItems: 'center',
              background: 'white',
              padding: '10px',
              justifyContent: 'space-between',
              height: '100%'
            }}
          >
            <Header
              style={{
                display: 'flex',
                alignItems: 'center',
                background: 'white',
                padding: '10px',
                justifyContent: 'space-between',
                width: '100%'
              }}
            >
              <div>源</div>
              <Radio.Group
                onChange={(e) => selectSourceType(e)}
                value={snap.sourceType}
                buttonStyle="solid"
                disabled={snap.processing}
              >
                <Radio.Button value="camera">摄像头</Radio.Button>
                <Radio.Button value="multimedia">视频/图片</Radio.Button>
              </Radio.Group>
            </Header>
            <Content
              style={{
                width: '100%',
                height: '100%',
                justifyContent: 'center',
                display: 'flex',
                alignItems: 'center'
              }}
            >
              {snap.sourceType === 'multimedia' && !Store.sourceFilePath && (
                <Button type="primary" onClick={() => chooseSourceFile()}>
                  选择文件
                </Button>
              )}
              {snap.sourceType === 'multimedia' &&
                snap.sourceFileType === 'video' &&
                snap.sourceFilePath && (
                  <ReactPlayer
                    width="100%"
                    height="100%"
                    url={`media-loader://${snap.sourceFilePath}`}
                    controls
                  />
                )}
              {snap.sourceType === 'multimedia' &&
                snap.sourceFileType === 'image' &&
                snap.sourceFilePath && (
                  <img width="100%" height="100%" src={`media-loader://${snap.sourceFilePath}`} />
                )}
              {snap.sourceType === 'camera' && <Camera />}
            </Content>
          </Layout>
        </Col>
        <Col
          span={12}
          style={{
            width: '100%',
            height: '100%',
            justifyContent: 'center',
            display: 'flex',
            alignItems: 'center',
            background: 'white'
          }}
        >
          <Progress
            type="circle"
            percent={snap.processProgress}
            strokeColor={{ '0%': '#108ee9', '100%': '#87d068' }}
          />
          {!snap.processing && <Button onClick={() => startProcess()}>开始</Button>}
          {snap.processing && <Button onClick={() => cancelProcess()}>取消</Button>}

          {snap.sourceType === 'camera' && snap.processing && <SwappedCamera></SwappedCamera>}
        </Col>
      </Row>
      <Row
        style={{
          height: '50%'
        }}
      >
        <Col span={12}>
          <div>参考图片</div>
          {snap.targetFilePath && (
            <img width="100%" height="100%" src={`media-loader://${snap.targetFilePath}`} />
          )}
          {!Store.targetFilePath && (
            <Button type="primary" onClick={() => chooseTargetFile()}>
              选择图片
            </Button>
          )}
        </Col>
        <Col span={12}>
          <div>配置:</div>
          <div>选择要替换的面部区域:</div>
          <Checkbox.Group style={{ width: '100%' }} onChange={onChangeConfig}>
            <Row>
              <Col span={8}>
                <Checkbox value="A">A</Checkbox>
              </Col>
              <Col span={8}>
                <Checkbox value="B">B</Checkbox>
              </Col>
              <Col span={8}>
                <Checkbox value="C">C</Checkbox>
              </Col>
              <Col span={8}>
                <Checkbox value="D">D</Checkbox>
              </Col>
              <Col span={8}>
                <Checkbox value="E">E</Checkbox>
              </Col>
            </Row>
          </Checkbox.Group>
        </Col>
      </Row>
    </div>
  )
}

export default FaceSwap
