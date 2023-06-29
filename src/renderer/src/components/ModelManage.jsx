import {
  Button,
  Checkbox,
  Col,
  Collapse,
  Form,
  Input,
  Layout,
  List,
  Menu,
  Progress,
  Radio,
  Row,
  Space,
  theme
} from 'antd'
import React, { useEffect, useState } from 'react'

import Store from '../Store'
import { useSnapshot } from 'valtio'

const { Panel } = Collapse

const ModelManage = () => {
  const snap = useSnapshot(Store)
  window.api.handleUpdateDownloadProgress((event, value) => {
    console.log('value======', value)
    Store.downloading = value
  })

  const chooseFolder = async () => {
    let filePaths = await window.api.chooseFileOrFolder(['openDirectory'])
    console.log('res', filePaths[0])
    window.api.store.set('modelFolder', filePaths[0])
    Store.modelFolder = filePaths[0]
  }

  const startDownload = () => {
    window.api.startDownload()
  }

  return (
    <div
      style={{
        height: '100%'
      }}
    >
      <div className="download-header">
        <div className="download-header-part1">
          模型路径:{snap.modelFolder}
          <Button type={snap.modelFolder ? 'link' : 'primary'} onClick={() => chooseFolder()}>
            {snap.modelFolder ? '重新选择' : '选择'}
          </Button>
        </div>
        {snap.modelFolder && (
          <Button type="primary" onClick={() => downloadAll()}>
            下载全部
          </Button>
        )}
      </div>

      <div>
        <div>
          <div>所需模型</div>
        </div>
        <div>
          <Collapse
            collapsible="header"
            defaultActiveKey={[0]}
            style={{ overflow: 'scroll', height: '100%' }}
          >
            {snap.models.map((item, index) => {
              return (
                <Panel header={item.label} key={index}>
                  <List
                    itemLayout="horizontal"
                    dataSource={item.files}
                    renderItem={(fileInfo, index) => (
                      <List.Item
                        key={index}
                        style={{ display: 'flex', alignItems: 'center', height: '50px' }}
                      >
                        <div
                          style={{
                            width: '80px'
                          }}
                        >
                          {fileInfo.fileName}
                        </div>
                        {fileInfo.downloading && (
                          <Progress
                            percent={fileInfo.processing}
                            strokeColor={{ '0%': '#108ee9', '100%': '#87d068' }}
                          />
                        )}
                        {!fileInfo.downloading && (
                          <Button type="primary" onClick={() => startDownload(fileInfo)}>
                            {fileInfo.downloaded ? '重新下载' : '下载'}
                          </Button>
                        )}
                      </List.Item>
                    )}
                  />
                </Panel>
              )
            })}
          </Collapse>
        </div>
      </div>
    </div>
  )
}

export default ModelManage
