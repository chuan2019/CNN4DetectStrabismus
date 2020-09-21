'use strict';
const video = document.querySelector('video');
let stream;










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