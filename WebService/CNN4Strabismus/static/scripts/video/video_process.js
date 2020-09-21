'use strict';

let utils = new Utils();

let streaming = false;
let videoInput = document.getElementById('videoInput');
let videoOnOff = document.getElementById('videoOnOff');
let canvasOutput = document.getElementById('canvasOutput');

const FPS = 30;

videoOnOff.addEventListener('click', () => {
    if (!streaming) {
        utils.startCamera('qvga', onVideoStarted, 'videoInput');
    } else {
        utils.stopCamera();
        onVideoStopped();
    }
});

function onVideoStarted() {
    streaming = true;
    videoOnOff.innerText = 'Close Camera';
    videoInput.width = videoInput.videoWidth;
    videoInput.height = videoInput.videoHeight;
    // schedule the first one.
    setTimeout(processVideo, 0);
}

function onVideoStopped() {
    streaming = false;
    canvasOutput.getContext('2d').clearRect(0, 0, canvasOutput.width, canvasOutput.height);
    videoOnOff.innerText = 'Open Camera';
}

utils.loadOpenCv(() => {
    let faceCascadeFile = 'haarcascade_frontalface_default.xml';
    utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
        videoOnOff.removeAttribute('disabled');
    })
});

function processVideo() {
    let src = new cv.Mat(videoInput.height, videoInput.width, cv.CV_8UC4);
    let dst = new cv.Mat(videoInput.height, videoInput.width, cv.CV_8UC4);
    let gray = new cv.Mat();
    let cap = new cv.VideoCapture(videoInput);
    let faces = new cv.RectVector();
    let classifier = new cv.CascadeClassifier();

    // load pre-trained classifiers
    classifier.load('haarcascade_frontalface_default.xml');

    try {
        if (!streaming) {
            // clean and stop.
            src.delete();
            dst.delete();
            gray.delete();
            faces.delete();
            classifier.delete();
            return;
        }
        let begin = Date.now();
        // start processing.
        cap.read(src);
        src.copyTo(dst);
        cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
        // detect faces.
        classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
        // draw faces.
        for (let i = 0; i < faces.size(); ++i) {
            let face = faces.get(i);
            let point1 = new cv.Point(face.x, face.y);
            let point2 = new cv.Point(face.x + face.width,
                                      face.y + face.height);
            cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
        }
        cv.imshow('canvasOutput', dst);
        // schedule the next one.
        let delay = 1000/FPS - (Date.now() - begin);
        setTimeout(processVideo, delay);
    } catch (err) {
        utils.printError(err);
    }
}
