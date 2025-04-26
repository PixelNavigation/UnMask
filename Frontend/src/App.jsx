import './App.css'
import Dropzone, { useDropzone } from 'react-dropzone'

function App() {
  return (
    <div>
      <div className='LinkUploader'>
        <h1>Paste your link to scan the video</h1>
        <input
          type='text'
          placeholder='Enter a link to upload'
          className='LinkUploader-input'
        />
        <button className='Link-button'>Upload</button>
      </div>
      <div className='VideoUploader'>
        <h1>Or, Upload the video directly</h1>
        <Dropzone
          onDrop={(acceptedFiles) => {
            console.log(acceptedFiles)
          }}
          accept={{ 'video/*': [] }}
          multiple={false}
          maxFiles={1}
          maxSize={100 * 1024 * 1024} // 100 MB
          noClick={true}
          noKeyboard={true}
        >
          {({ getRootProps, getInputProps }) => (
            <div {...getRootProps()} className='Dropzone'>
              <input {...getInputProps()} />
              <p>Drag and drop a video file here, or click to select a file</p>
            </div>
          )}
        </Dropzone>
      </div>
    </div>
  )
}

export default App
