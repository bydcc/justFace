import { Button, Progress, Tooltip } from 'antd'
import { useEffect, useState } from 'react'

import Store from '../Store'
import { TranslationOutlined } from '@ant-design/icons'
import { useSnapshot } from 'valtio'
import { useTranslation } from 'react-i18next'

const Home = () => {
  const snap = useSnapshot(Store)
  const { t, i18n } = useTranslation()
  window.api.handleUpdateDownloadProgress((event, fileName, percent) => {
    Store.downloadProgress = percent
    Store.downloadFileName = fileName
    if (percent === 1) {
      Store.modelDownloaded = true
    }
  })
  useEffect(() => {
    Store.modelFolder = window.api.store.get('modelFolder')
    Store.modelDownloaded = window.api.store.get('modelDownloaded')
  })

  const changeLanguage = () => {
    i18n.changeLanguage(i18n.resolvedLanguage === 'zh' ? 'en' : 'zh')
  }

  const downloadModel = () => {
    window.api.startDownload()
  }

  return (
    <div
      style={{
        height: '100%'
      }}
    >
      <Tooltip title="中文/English">
        <Button onClick={changeLanguage} shape="circle" icon={<TranslationOutlined />} />
      </Tooltip>
      <div>
        <div></div>
        <div>
          <div>xxxxx</div>
          <div>使用说明</div>
          <div>step1:</div>
          <div>step2:</div>
        </div>
        {!snap.modelDownloaded && (
          <div>
            模型文件不全，请先
            <Button type="primary" onClick={() => downloadModel()}>
              下载模型
            </Button>
          </div>
        )}
        {snap.downloadProgress !== 1 && (
          <div>
            <div>正在下载:{snap.downloadFileName}</div>
            <Progress
              percent={snap.downloadProgress * 100}
              showInfo={false}
              strokeColor={{ '0%': '#108ee9', '100%': '#87d068' }}
            />
          </div>
        )}
      </div>
    </div>
  )
}

export default Home
