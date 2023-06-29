import { BrowserWindow, app, dialog, ipcMain, protocol, screen, shell } from 'electron'
import { cleanUpPython, runPython } from './runPy'
import { electronApp, is, optimizer } from '@electron-toolkit/utils'

import { getModelsToDownload } from './modelInfo'
import icon from '../../resources/icon.png?asset'
import { io } from 'socket.io-client'
import { join } from 'path'

const { download } = require('electron-dl')

const PersistentStore = require('electron-store')
let persistentStore = new PersistentStore()
const appPath = app.getAppPath()
console.log('appPath', appPath)
const url = require('url')
protocol.registerSchemesAsPrivileged([{ scheme: 'media-loader', privileges: { bypassCSP: true } }])

function createWindow() {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 900,
    height: 670,
    show: false,
    autoHideMenuBar: true,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    },
    alwaysOnTop: false
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
    mainWindow.webContents.openDevTools()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }

  return mainWindow
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(async () => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.electron')

  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  // see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  // Create custom protocol for local media loading
  protocol.registerFileProtocol('media-loader', (request, callback) => {
    const filePath = url.fileURLToPath('file://' + request.url.slice('media-loader://'.length))

    try {
      return callback(filePath)
    } catch (err) {
      console.error(err)
      return callback(404)
    }
  })

  // Check model all downloaded
  let toDownload = await getModelsToDownload(appPath)
  console.log('toDownload', toDownload, toDownload.length === 0)
  persistentStore.set('modelDownloaded', toDownload.length === 0)

  const mainWindow = createWindow()

  const socket = io('ws://localhost:8000', {
    path: '/sio/socket.io/',
    autoConnect: false
  })
  socket.on('progress.update', (data) => {
    // 将接收到的进度信息发送给渲染进程
    mainWindow.webContents.send('update-progress', data.toString())
  })

  socket.connect()
  ipcMain.handle('choose-file-or-folder', async (event, properties, filters) => {
    console.log('aaaaa', properties)
    const { filePaths } = await dialog.showOpenDialog({
      properties: properties,
      filters: filters
    })
    return filePaths
  })

  ipcMain.handle('start-process', (event, data) => {
    socket.emit('progress.start', data)
  })

  ipcMain.handle('cancel-process', () => {
    socket.emit('progress.cancel')
  })

  ipcMain.handle('release-camera', (event) => {
    socket.emit('camera.release')
  })

  // 监听渲染进程发出的download事件
  ipcMain.handle('start-download', async (event) => {
    let toDownload = await getModelsToDownload(appPath)
    console.log('toDownload======', toDownload)
    for (let index in toDownload) {
      await download(mainWindow, toDownload[index].url, {
        directory: join(appPath, toDownload[index].filePath),
        filename: toDownload[index].fileName,
        onTotalProgress: (res) => {
          console.log('res111', res)
          mainWindow.webContents.send(
            'update-download-progress',
            toDownload[index].fileName,
            res.percent
          )
        }
      })
    }
  })

  ipcMain.on('electron-store-get', async (event, val) => {
    event.returnValue = persistentStore.get(val)
  })
  ipcMain.on('electron-store-set', async (event, key, val) => {
    persistentStore.set(key, val)
  })

  // app.on('activate', function () {
  //   // On macOS it's common to re-create a window in the app when the
  //   // dock icon is clicked and there are no other windows open.
  //   if (BrowserWindow.getAllWindows().length === 0) createWindow()
  // })
})

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

// In this file you can include the rest of your app"s specific main process
// code. You can also put them in separate files and require them here.
app.on('ready', (appPath) => {
  runPython(appPath)
})
app.on('will-quit', cleanUpPython)
