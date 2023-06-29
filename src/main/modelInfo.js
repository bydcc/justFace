const fs = require('fs')
const path = require('path')

const models = [
  {
    key: 1,
    url: 'https://drive.google.com/uc?id=1cyJTYRO5G4kcugAcgSJ7cMsE96GzV_hq&export=download&confirm=9iBg',
    fileName: 'iteration_300000.pt',
    filePath: 'pretrained_ckpts/e4s'
  },
  {
    key: 2,
    url: 'https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812&export=download',
    fileName: '79999_iter.pth',
    filePath: 'pretrained_ckpts/face_parsing'
  }
  // {
  //   key: 3,
  //   url: 'xxx',
  //   fileName: 'segnext.tiny.best_mIoU_iter_160000.pth',
  //   filePath: 'pretrained_ckpts/face_parsing'
  // },
  // {
  //   key: 4,
  //   url: 'xxxx',
  //   fileName: '00000189-checkpoint.pth.tar',
  //   filePath: 'pretrained_ckpts/facevid2vid'
  // },
  // {
  //   key: 5,
  //   url: 'xxxxxxxxxxxxxxxx',
  //   fileName: 'vox-256.yaml',
  //   filePath: 'pretrained_ckpts/facevid2vid'
  // },
  // { key: 6, url: 'xxxxx', fileName: 'GPEN-BFR-512.pth', filePath: 'gen/weights' },
  // { key: 7, url: 'xxxxx', fileName: 'ParseNet-latest.pth', filePath: 'gen/weights' },
  // { key: 8, url: 'xxxxxx', fileName: 'realesrnet_x4.pth', filePath: 'gen/weights' },
  // { key: 9, url: 'xxxxx', fileName: 'RetinaFace-R50.pth', filePath: 'gen/weights' }
]

const checkFilesExist = (filePaths) => {
  const promises = filePaths.map((filePath) => {
    return new Promise((resolve) => {
      fs.access(filePath, (err) => {
        if (err) {
          resolve(false)
        } else {
          resolve(true)
        }
      })
    })
  })

  return Promise.all(promises)
}

function isFileAccessible(filePath) {
  return new Promise((resolve, reject) => {
    fs.access(filePath, (err) => {
      if (err) {
        reject(err)
      } else {
        resolve(true)
      }
    })
  })
}

const getModelsToDownload = async (appPath) => {
  let toDownload = []
  for (let index in models) {
    let filePath = path.join(appPath, models[index].filePath, models[index].fileName)
    try {
      await isFileAccessible(filePath)
      console.log(`文件 ${filePath} 可访问`)
    } catch (err) {
      toDownload.push(models[index])
    }
  }
  console.log('toDownload', toDownload)
  return toDownload
}

export { models, getModelsToDownload }
