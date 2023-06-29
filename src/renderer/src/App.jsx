import { CloudDownloadOutlined, HomeOutlined, SmileOutlined } from '@ant-design/icons'
import { Layout, Menu, theme } from 'antd'

import FaceSwap from './components/FaceSwap'
import Home from './components/Home'
import ModelManage from './components/ModelManage'
import React from 'react'
import Store from './Store'
import { useSnapshot } from 'valtio'
import { useTranslation } from 'react-i18next'

const { Header, Content, Footer, Sider } = Layout

const App = () => {
  const snap = useSnapshot(Store)
  const { t } = useTranslation()
  const items = [
    {
      key: 'home',
      icon: React.createElement(HomeOutlined),
      label: t('home')
    },
    {
      key: 'swap',
      icon: React.createElement(SmileOutlined),
      label: t('swapFace')
    }
    // {
    //   key: 'model',
    //   icon: React.createElement(CloudDownloadOutlined),
    //   label: t('model')
    // }
  ]
  const {
    token: { colorBgContainer }
  } = theme.useToken()

  const changeMenu = (e) => {
    Store.menu = e.key
  }

  return (
    <Layout hasSider>
      <Sider
        style={{
          overflow: 'auto',
          height: '100vh',
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0
        }}
      >
        <div className="demo-logo-vertical" />
        <Menu
          theme="dark"
          mode="inline"
          defaultSelectedKeys={['extract']}
          items={items}
          onSelect={(e) => changeMenu(e)}
        />
      </Sider>
      <Layout
        className="site-layout"
        style={{
          marginLeft: 200,
          height: '100vh'
        }}
      >
        <Header
          style={{
            padding: 0,
            background: colorBgContainer
          }}
        />
        <Content
          style={{
            margin: '24px 16px 0',
            overflow: 'initial'
          }}
        >
          <div
            style={{
              width: '100%',
              height: '100%',
              display: snap.menu === 'swap' ? 'block' : 'none'
            }}
          >
            <FaceSwap />
          </div>
          {/* <div
            style={{
              width: '100%',
              height: '100%',
              display: snap.menu === 'model' ? 'block' : 'none'
            }}
          >
            <ModelManage />
          </div> */}
          <div
            style={{
              width: '100%',
              height: '100%',
              display: snap.menu === 'home' ? 'block' : 'none'
            }}
          >
            <Home />
          </div>
        </Content>
        <Footer
          style={{
            textAlign: 'center'
          }}
        >
          Ant Design ©2023 Created by Ant UED
        </Footer>
      </Layout>
    </Layout>
  )
}
export default App
