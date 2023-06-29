import { Button, Checkbox, Form, Input, Menu, Space, theme } from 'antd'
import { useRef, useState } from 'react'

import React from 'react'

const FaceExtract = () => {
  const [form] = Form.useForm()
  const chooseFolder = async (name) => {
    let filePaths = await window.api.chooseFileOrFolder(['openDirectory'])
    console.log('res', filePaths[0])
    form.setFieldValue(name, filePaths[0])
  }

  const onFinish = (values) => {
    console.log('Success:', values)
  }

  const onFinishFailed = (errorInfo) => {
    console.log('Failed:', errorInfo)
  }
  return (
    <div>
      <Form
        form={form}
        labelCol={{ span: 8 }}
        wrapperCol={{ span: 16 }}
        style={{ maxWidth: 600 }}
        onFinish={onFinish}
        onFinishFailed={onFinishFailed}
        autoComplete="off"
      >
        <Form.Item label="读取于">
          <Space>
            <Form.Item
              noStyle
              name="source"
              rules={[{ required: true, message: '原视频或图片所在的文件夹' }]}
            >
              <Input />
            </Form.Item>
            <Button onClick={() => chooseFolder('source')}>快速选择</Button>
          </Space>
        </Form.Item>
        <Form.Item label="存储到">
          <Space>
            <Form.Item
              noStyle
              name="target"
              rules={[{ required: true, message: '提取的人脸图片存放的文件夹' }]}
            >
              <Input />
            </Form.Item>
            <Button onClick={() => chooseFolder('target')}>快速选择</Button>
          </Space>
        </Form.Item>
        <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
          <Button type="primary" htmlType="submit">
            Submit
          </Button>
        </Form.Item>
      </Form>
    </div>
  )
}

export default FaceExtract
