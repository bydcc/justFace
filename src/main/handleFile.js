const fs = require('fs')

const checkFileExists = (path) => {
  // 检测文件是否存在
  fs.access(path, fs.constants.F_OK, (err) => {
    if (err) {
      return false
    } else {
      return true
    }
  })
}

export { checkFileExists }
