'use strict';

const constraints = window.constraints = {
    audio: false,
    video: true
};

const video = document.querySelector('video');

function handleSuccess(stream) {
    const videoTracks = stream.getVideoTracks();
    console.log('Got stream with constraints:', constraints);
    console.log(`Using video device: ${videoTracks[0].label}`);
    window.stream = stream; // make variable available to browser console
    video.srcObject = stream;
}


function handleError(error) {
    if (error.name === 'ConstraintNotSatisfiedError') {
        const v = constraints.video;
        errorMsg(`The resolution ${v.width.exact}x${v.height.exact} px is not supported by your device.`);
    } else if (error.name === 'PermissionDeniedError') {
        errorMsg('Permissions have not been granted to use your camera and ' +
            'microphone, you need to allow the page access to your devices in ' +
            'order for the demo to work.');
    }
    errorMsg(`getUserMedia error: ${error.name}`, error);
}
  
function errorMsg(msg, error) {
    const errorElement = document.querySelector('#errorMsg');
    errorElement.innerHTML += `<p>${msg}</p>`;
    if (typeof error !== 'undefined') {
        console.error(error);
    }
}

async function init(e) {
    if (e.target.value == "Open Camera") {
        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            handleSuccess(stream);
            e.target.value = "Close Camera";
        } catch (e) {
            handleError(e);
        }
    } else {

    }
}

video.addEventListener('click', e => init(e));








function capture_video() {
    // Put variables in global scope to make them available to the browser console.
    const video = document.querySelector('video');
    const canvas = window.canvas = document.querySelector('canvas');
    canvas.width = 300;
    canvas.height = 280;

    const button = document.querySelector('button');
    button.onclick = function() {
        console.log('Take snapshot button clicked!');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    };

    const constraints = {
        audio: false,
        video: true
    };

    function handleSuccess(stream) {
    window.stream = stream; // make stream available to browser console
        video.srcObject = stream;
    }

    function handleError(error) {
        console.log('navigator.MediaDevices.getUserMedia error: ', error.message, error.name);
    }

    navigator.mediaDevices.getUserMedia(constraints).then(handleSuccess).catch(handleError);
}