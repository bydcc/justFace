const { spawn } = require('child_process')
const fs = require('fs')
const path = require('path')
const os = require('os')

const homeDir = os.homedir()

// 检查是否已经安装 pyenv
const pyenvDir = path.join(homeDir, '.pyenv')

let pyenvProcess
let virtualEnvProcess
let pythonProcess

const runPython = (appPath) => {
  // 检查是否已经安装 pyenv
  if (!fs.existsSync(pyenvDir)) {
    // 如果尚未安装 pyenv，则执行安装步骤
    // 安装 pyenv
    pyenvProcess = spawn('curl', ['https://pyenv.run', '-sSf', '|', 'bash'])

    pyenvProcess.stdout.on('data', (data) => {
      console.log(`stdout: ${data}`)
    })

    pyenvProcess.stderr.on('data', (data) => {
      console.error(`stderr: ${data}`)
    })

    pyenvProcess.on('close', (code) => {
      console.log(`child process exited with code ${code}`)

      // 创建虚拟环境
      createVirtualEnv(appPath)
    })
  } else {
    // 如果已经安装 pyenv，则跳过安装步骤，直接创建虚拟环境
    createVirtualEnv(appPath)
  }
}

function createVirtualEnv(appPath) {
  // 检查虚拟环境是否存在
  const virtualEnvDir = path.join(homeDir, '.pyenv', 'versions', 'e4senv')
  if (!fs.existsSync(virtualEnvDir)) {
    // 如果虚拟环境不存在，则执行创建步骤

    // 创建虚拟环境
    virtualEnvProcess = spawn('pyenv', ['virtualenv', 'e4senv'])

    virtualEnvProcess.stdout.on('data', (data) => {
      console.log(`stdout: ${data}`)
    })

    virtualEnvProcess.stderr.on('data', (data) => {
      console.error(`stderr: ${data}`)
    })

    virtualEnvProcess.on('close', (code) => {
      console.log(`child process exited with code ${code}`)

      // 激活虚拟环境
      activateVirtualEnv(appPath)
    })
  } else {
    // 如果虚拟环境已经存在，则跳过创建步骤，直接激活虚拟环境
    activateVirtualEnv(appPath)
  }
}

function activateVirtualEnv(appPath) {
  // 激活虚拟环境
  const activateCmd =
    os.platform() === 'win32' ? 'activate e4senv' : 'source ~/.pyenv/versions/e4senv/bin/activate'
  const pyenvActivate = spawn(activateCmd, [], { shell: true })

  pyenvActivate.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`)
  })

  pyenvActivate.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`)
  })

  pyenvActivate.on('close', (code) => {
    console.log(`child process exited with code ${code}`)

    // 安装 requirements.txt 中的包
    const pipInstall = spawn('pip', [
      'install',
      '-r',
      path.join(appPath, 'python_backend', 'requirements.txt')
    ])

    pipInstall.stdout.on('data', (data) => {
      console.log(`stdout: ${data}`)
    })

    pipInstall.stderr.on('data', (data) => {
      console.error(`stderr: ${data}`)
    })

    pipInstall.on('close', (code) => {
      console.log(`child process exited with code ${code}`)

      // 运行 Python 代码
      pythonProcess = spawn('python', ['myscript.py'])

      pythonProcess.stdout.on('data', (data) => {
        console.log(`stdout: ${data}`)
      })

      pythonProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`)
      })

      pythonProcess.on('close', (code) => {
        console.log(`child process exited with code ${code}`)
      })
    })
  })
}

const cleanUpPython = () => {
  if (pyenvProcess) {
    pyenvProcess.kill()
  }
  if (virtualEnvProcess) {
    virtualEnvProcess.kill()
  }
  if (pythonProcess) {
    pythonProcess.kill()
  }
}

export { runPython, cleanUpPython }
