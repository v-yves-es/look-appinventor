// -*- mode: javascript; js-indent-level: 2; -*-
"use strict";

console.log("Look: Using TensorFlow.js version " + tf.version.tfjs);

// AANGEPAST: gebruik lokale modelbestanden in plaats van externe URL
const MOBILENET_MODEL_PATH = "./web_model.pb";
const WEIGHTS_MANIFEST_FILE_URL = "./weights_manifest.json";

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;

// make sure error codes are consistent with those defined in Look.java
const ERROR_CLASSIFICATION_NOT_SUPPORTED = -1;
const ERROR_CLASSIFICATION_FAILED = -2;
const ERROR_CANNOT_TOGGLE_CAMERA_IN_IMAGE_MODE = -3;
const ERROR_CANNOT_CLASSIFY_IMAGE_IN_VIDEO_MODE = -4;
const ERROR_CANNOT_CLASSIFY_VIDEO_IN_IMAGE_MODE = -5;
const ERROR_INVALID_INPUT_MODE = -6;

let mobilenet;
const mobilenetDemo = async () => {
  try {
    mobilenet = await tf.loadFrozenModel(MOBILENET_MODEL_PATH, WEIGHTS_MANIFEST_FILE_URL);
    const zeros = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    mobilenet.predict(zeros).dispose();
    zeros.dispose();
    console.log("Look: Mobilenet ready");
    // AANGEPAST: Look.ready() bestaat niet in standalone mode, stuur naar bridge
    if (window.Look && typeof window.Look.ready === "function") {
      Look.ready();
    } else {
      console.log("Look: Model ready (standalone mode, no Android bridge)");
    }
  } catch (error) {
    console.log("Look: " + error);
    // AANGEPAST: controleer of Look.error bestaat
    if (window.Look && typeof window.Look.error === "function") {
      Look.error(ERROR_CLASSIFICATION_NOT_SUPPORTED);
    } else {
      console.error("Look: Model load failed (no Android bridge to report error)");
    }
  }
};

async function predict(pixels) {
  try {
    const logits = tf.tidy(() => {
      const img = tf.image.resizeBilinear(tf.fromPixels(pixels).toFloat(), [IMAGE_SIZE, IMAGE_SIZE]);
      const offset = tf.scalar(127.5);
      const normalized = img.sub(offset).div(offset);
      const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
      return mobilenet.predict(batched);
    });
    const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
    logits.dispose();
    var result = [];
    for (let i = 0; i < classes.length; i++) {
      result.push([classes[i].className, classes[i].probability.toFixed(5)]);
    }
    console.log("Look: prediction is " + JSON.stringify(result));

    // --- NEW: stuur voorspellingen ook naar de App Inventor WebViewer bridge indien aanwezig
    try {
      // zet om naar formaat [{label, confidence}, ...] met confidence als number
      const preds = result.map(function(r){ return { label: r[0], confidence: parseFloat(r[1]) }; });
      if (window.onPredictionsReady && typeof window.onPredictionsReady === "function") {
        try { window.onPredictionsReady(preds); } catch(e) { console.warn("onPredictionsReady error:", e); }
      }
    } catch(e) {
      console.warn("Could not send predictions to App Inventor bridge:", e);
    }

    // Bestaande Android callback (houd dit zodat Android-extensie blijft werken)
    if (window.Look && typeof window.Look.reportResult === "function") {
      Look.reportResult(JSON.stringify(result));
    }
  } catch (error) {
    console.log("Look: " + error);
    if (window.Look && typeof window.Look.error === "function") {
      Look.error(ERROR_CLASSIFICATION_NOT_SUPPORTED);
    }
  }
}

async function getTopKClasses(logits, topK) {
  const values = await logits.data();
  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }
  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: SCAVENGER_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    });
  }
  return topClassesAndProbs;
}

var img = document.createElement("img");
img.width = window.innerWidth;
img.style.display = "block";

var video = document.createElement("video");
video.setAttribute("autoplay", "");
video.setAttribute("playsinline", "");
video.width = window.innerWidth;
video.style.display = "none";

var frontFacing = false;
var isVideoMode = false;

document.body.appendChild(img);
document.body.appendChild(video);

video.addEventListener("loadedmetadata", function() {
  video.height = this.videoHeight * video.width / this.videoWidth;
}, false);

function startVideo() {
  if (isVideoMode) {
    navigator.mediaDevices.getUserMedia({video: {facingMode: frontFacing ? "user" : "environment"}, audio: false})
    .then(stream => (video.srcObject = stream))
    .catch(e => console.error("startVideo error:", e));
    video.style.display = "block";
  }
}

function stopVideo() {
  if (isVideoMode && video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.style.display = "none";
  }
}

function toggleCameraFacingMode() {
  if (isVideoMode) {
    frontFacing = !frontFacing;
    stopVideo();
    startVideo();
  } else {
    if (window.Look && typeof window.Look.error === "function") {
      Look.error(ERROR_CANNOT_TOGGLE_CAMERA_IN_IMAGE_MODE);
    }
  }
}

function classifyImageData(imageData) {
  if (!isVideoMode) {
    img.onload = function() {
      predict(img);
    }
    img.src = "data:image/png;base64," + imageData;
  } else {
    if (window.Look && typeof window.Look.error === "function") {
      Look.error(ERROR_CANNOT_CLASSIFY_IMAGE_IN_VIDEO_MODE);
    }
  }
}

function classifyVideoData() {
  if (isVideoMode) {
    predict(video);
  } else {
    if (window.Look && typeof window.Look.error === "function") {
      Look.error(ERROR_CANNOT_CLASSIFY_VIDEO_IN_IMAGE_MODE);
    }
  }
}

function setInputMode(inputMode) {
  if (inputMode === "image" && isVideoMode) {
    stopVideo();
    isVideoMode = false;
    img.style.display = "block";
  } else if (inputMode === "video" && !isVideoMode) {
    img.style.display = "none";
    isVideoMode = true;
    startVideo();
  } else if (inputMode !== "image" && inputMode !== "video") {
    if (window.Look && typeof window.Look.error === "function") {
      Look.error(ERROR_INVALID_INPUT_MODE);
    }
  }
}

window.addEventListener("resize", function() {
  img.width = window.innerWidth;
  video.width = window.innerWidth;
  video.height = video.videoHeight * window.innerWidth / video.videoWidth;
});

// --- NEW: convenience wrappers voor gebruik vanuit WebViewer / console / bridge

// Classificeer een Image DOM element (bv. <img> of canvas)
window.classifyImageElement = function(imgElement) {
  // retourneert een promise die wordt vervuld zodra predict klaar is (indien predict async)
  return predict(imgElement);
};

// Classificeer een data URL ("data:image/jpeg;base64,...") of pure base64 string.
// Als je een volledige data URL meegeeft, wordt het juiste deel eruit gehaald.
window.classifyDataUrl = function(dataUrlOrBase64) {
  if (!dataUrlOrBase64) return;
  if (dataUrlOrBase64.indexOf('data:') === 0) {
    var comma = dataUrlOrBase64.indexOf(',');
    var b64 = dataUrlOrBase64.substring(comma + 1);
    classifyImageData(b64);
  } else {
    // verondersteld pure base64 zonder prefix
    classifyImageData(dataUrlOrBase64);
  }
};

// Classificeer een frame uit de video (indien video mode actief)
window.classifyVideo = function() {
  classifyVideoData();
};

// Als de pagina in een App Inventor WebViewer draait en die WebViewer de bridge configureert,
// moet die bridge window.onPredictionsReady(preds) definiëren (dat doet de aangepaste look_appinventor_bridge.html).

mobilenetDemo();