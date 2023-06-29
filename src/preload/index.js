import { contextBridge, ipcRenderer } from 'electron'

import { electronAPI } from '@electron-toolkit/preload'

// Custom APIs for renderer
const api = {
  chooseFileOrFolder: async (properties, filters) => {
    return await ipcRenderer.invoke('choose-file-or-folder', properties, filters)
  },

  startProcess: (data) => {
    ipcRenderer.invoke('start-process', data)
  },

  cancelProcess: (data) => {
    ipcRenderer.invoke('cancel-process', data)
  },

  handleUpdateProgress: (callback) => ipcRenderer.on('update-progress', callback),

  handleUpdateDownloadProgress: (callback) => ipcRenderer.on('update-download-progress', callback),

  startDownload: () => {
    ipcRenderer.invoke('start-download')
  },

  releaseCamera: () => {
    ipcRenderer.invoke('release-camera')
  },

  checkFileExists: (fileUrls) => {
    return ipcRenderer.invoke('check-file-exists', fileUrls)
  },

  store: {
    get(key) {
      return ipcRenderer.sendSync('electron-store-get', key)
    },
    set(property, val) {
      ipcRenderer.send('electron-store-set', property, val)
    }
  }
}

// Use `contextBridge` APIs to expose Electron APIs to
// renderer only if context isolation is enabled, otherwise
// just add to the DOM global.
if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('api', api)
  } catch (error) {
    console.error(error)
  }
} else {
  window.electron = electronAPI
  window.api = api
}
