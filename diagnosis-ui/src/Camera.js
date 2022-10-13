import React from 'react';
import CameraPhoto, { FACING_MODES } from 'jslib-html5-camera-photo';
import IconButton from '@material-ui/core/IconButton';
import AddAPhotoIcon from '@material-ui/icons/AddAPhoto';
import {  Grid } from '@material-ui/core';
import axios from 'axios';

class Camera extends React.Component {
  constructor(props, context) {
    super(props, context);
    this.cameraPhoto = null;
    this.videoRef = React.createRef();
    this.state = {
      dataUri: 'diagnosis.png',
      isCamStart: false
    }
  }

  componentDidMount() {
    this.cameraPhoto = new CameraPhoto(this.videoRef.current);
  }

  startCamera(idealFacingMode, idealResolution) {
    console.log('start Camera');
    this.cameraPhoto.startCamera(idealFacingMode, idealResolution)
      .then(() => {
        console.log('camera is started !');
      })
      .catch((error) => {
        console.error('Camera not started!', error);
      });
  }

  startCameraMaxResolution(idealFacingMode) {
    this.cameraPhoto.startCameraMaxResolution(idealFacingMode)
      .then(() => {
        console.log('camera is started !');
      })
      .catch((error) => {
        console.error('Camera not started!', error);
      });
  }

  takePhoto() {
    const config = {
      sizeFactor: 1
    };
    let dataUri = this.cameraPhoto.getDataUri(config);
    this.setState(ps => ({ ...ps, dataUri }));
    
    axios.post('https://diagnosis-test.herokuapp.com/snippets/', {
      'img': dataUri.substr(22)
    }).
      then(function (response) {
        console.log(response);
      })
      .catch(function (error) {
        console.log('axios error' + error);
      });
  }

  stopCamera() {
    console.log('stop Camera');
    this.cameraPhoto.stopCamera()
      .then(() => {
        console.log('Camera stoped!');
      })
      .catch((error) => {
        console.log('No camera to stop!:', error);
      });
  }

  onPhotoClick() {
    if (!this.state.isCamStart) {
      let isCamStart = !this.state.isCamStart;
      this.setState(ps => ({ ...ps, isCamStart }),
        () => {
          console.log(this.state.isCamStart)
        }
      );
      this.startCamera(FACING_MODES.ENVIRONMENT, { width: 5, height: 5 });

    }
    else {
      let isCamStart = !this.state.isCamStart;
      this.setState(ps => ({ ...ps, isCamStart }),
        () => {
          console.log(this.state.isCamStart)
        }
      );
      this.takePhoto();
      this.stopCamera();
    }
  }

  onAnalyse() {
  }

  render() {
    return (
      <div>
        <div>
          <video style={{ display: this.state.isCamStart ? '' : 'none' }} ref={this.videoRef} autoPlay={true} />
          <img
            alt=''
            src={this.state.dataUri}
            width='320px'
            height='235px'
            style={{ display: this.state.isCamStart ? 'none' : '' }}
          />
          {/* <Button variant="outlined" color="primary" >Analyse</Button> */}
        </div>
        <Grid>
          <Grid item>
            <IconButton onClick={() => { this.onPhotoClick(); }}>
              <AddAPhotoIcon />
            </IconButton>
          </Grid>
        </Grid>

      </div>
    );
  }
}

export default Camera;