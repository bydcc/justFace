import { proxy } from 'valtio'
const Store = proxy({
  menu: 'home',
  sourceType: 'multimedia',
  sourceFilePath: '',
  sourceFileType: '',
  targetFilePath: '',
  modelFolderPath: '',
  processing: false,
  processProgress: 0,
  language: 'zh',
  modelDownloaded: false,
  downloadProgress: 1,
  downloadFileName: ''
})

export default Store
