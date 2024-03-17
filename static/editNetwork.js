function generateLayerHTML(layerId, layerName) {
    return `
        <div aria-grabbed="false" id="${layerId}"
        class="navMargin flex items-center gap-3 rounded bg-grey-700 padx-3 pady-2 text-grey transition-all hover:text-white cursor-move"
        onclick="selectLayer('${layerId}')">

        <!-- Drag and Drop Layer Icon -->
        <svg class="handle h-4 w-4 text-grey" xmlns="http://www.w3.org/2000/svg" width="24" height="24"
            viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
            stroke-linecap="round" stroke-linejoin="round">
            <line x1="4" x2="20" y1="12" y2="12"></line>
            <line x1="4" x2="20" y1="6" y2="6"></line>
            <line x1="4" x2="20" y1="18" y2="18"></line>
        </svg>

        <!-- Layer Name -->
        ${layerName}

        <!-- Layer Status -->
        <span class="ml-auto bg-green-500 text-dark-grey padx-2 pady-1 rounded-full text-xs"
            id="validationStatus">Valid</span>
        </div>`;
}

function loadNetwork(layers) {
    const layersView = document.getElementById("layersView");
    layersView.innerHTML = Object.entries(layers)
        .map(([layerId, { layerName }]) => generateLayerHTML(layerId, layerName))
        .join('');

    validateLayers();
    saveLayers();
}

function addLayer(layerId, layerName, layerType) {
    layers[layerId] = {
        "layerName": layerName,
        "layerConfig": {
            // Default layer configuration values
            "inputSize": 0,
            "outputSize": 0,
            "activation": "sigmoid",
            "type": layerType
        },
        "valid": false,
        "errors": [],
        "status": "inactive",
    };

    // Load layer configuration
    loadNetwork(layers);
}

function selectLayer(layerId) {
    console.log("Selected layer: " + layerId);

    loadLayerConfiguration(layerId);

    // Change selected layer text colour
    for (const layer of document.getElementById("layersView").children) {
        const layerSelected = layer.id == layerId;
        layer.classList.toggle("text-white", layerSelected);
        layer.classList.toggle("text-grey", !layerSelected);
    }
}

function validateLayers() {
    Object.entries(layers).forEach(([layerId, layer]) => {
        const idNum = parseInt(layerId.split("-")[1]);
        const { layerConfig: { inputSize, outputSize } } = layer;
        layer.valid = true;
        layer.errors = [];

        // Check inputSize
        if (parseInt(inputSize) <= 0 || !Number.isInteger(parseInt(inputSize))) {
            layer.valid = false;
            layer.errors.push("Input size must be a positive integer");
        }

        const prevLayer = layers[`layer-${idNum - 1}`];
        const nextLayer = layers[`layer-${idNum + 1}`];

        if (idNum == 1 && inputSize != initialInputSize) {
            layer.valid = false;
            layer.errors.push(`Input size must match network input size of ${initialInputSize}`);
        } else if (prevLayer && inputSize != prevLayer.layerConfig.outputSize) {
            layer.valid = false;
            layer.errors.push(`Input size must match previous layer's output size of ${prevLayer.layerConfig.outputSize}`);
        }

        // Check outputSize
        if (parseInt(outputSize) <= 0 || !Number.isInteger(parseInt(outputSize))) {
            layer.valid = false;
            layer.errors.push("Output size must be a positive integer");
        }

        if (nextLayer && outputSize != nextLayer.layerConfig.inputSize) {
            layer.valid = false;
            layer.errors.push(`Output size must match next layer's input size of ${nextLayer.layerConfig.inputSize}`);
        } else if (!nextLayer && outputSize != initialOutputSize) {
            layer.valid = false;
            layer.errors.push(`Output size must match network output size of ${initialOutputSize}`);
        }

        // Update validation status
        let validationStatus = document.getElementById(layerId).querySelector("#validationStatus");
        validationStatus.innerHTML = layer.valid ? "Valid" : "Invalid";
        validationStatus.classList.toggle("bg-green-500", layer.valid);
        validationStatus.classList.toggle("bg-red-500", !layer.valid);
    });
}

function updateLayerOrder(event) {

    // Move layer in layers object
    const oldLayerId = "layer-" + (event.oldIndex + 1);
    const newLayerId = "layer-" + (event.newIndex + 1);

    const layer = layers[oldLayerId];
    delete layers[oldLayerId];

    // Rename keys to be sequential
    let count = 1;
    const newLayers = {};
    for (const id in layers) {
        if (('layer-' + count) == newLayerId) {
            newLayers['layer-' + count++] = layer;
            newLayers['layer-' + count++] = layers[id];
        } else {
            newLayers['layer-' + count++] = layers[id];
        }
    }

    if (!(newLayerId in newLayers)) {
        newLayers[newLayerId] = layer;
    }

    layers = newLayers;

    loadNetwork(layers);
}

function togglePopup(elementId) {
    const popup = document.getElementById(elementId);
    const isHidden = popup.style.display === "none";

    popup.style.display = isHidden ? "block" : "none";
    document.getElementById("newLayerPopupAddButton").disabled = isHidden;

    if (!isHidden) {
        for (const button of document.getElementsByClassName("layer-button")) {
            button.classList.remove("selected");
        }
    }
}

function selectLayerType(layerName, layerType) {
    // Change the colour of the selected button
    const layerButton = document.getElementById(`newLayerSelection-${layerType}`);
    layerButton.classList.add("selected");

    // Change the colour of the other buttons
    for (const button of document.getElementsByClassName("layer-button")) {
        if (button.id != layerButton.id) {
            button.classList.remove("selected");
        }
    }

    // Determine the layerId
    const layerId = "layer-" + (Object.keys(layers).length + 1);

    // Change the onclick function of the add button to add the selected layer type
    const addButton = document.getElementById("newLayerPopupAddButton");
    addButton.onclick = function () {
        // Close and reset the popup
        togglePopup("newLayerPopup");

        // Add the layer
        addLayer(layerId, layerName, layerType);
    };

    // Activate the add button
    addButton.disabled = false;
}

function saveLayerConfiguration(layerId) {
    event.preventDefault();

    const inputSize = document.getElementById("inputSize").value;
    const outputSize = document.getElementById("outputSize").value;
    const activation = document.getElementById("activation").value;

    if (layerId == "inputLayer" || layerId == "outputLayer") {
        if (layerId == "inputLayer") initialInputSize = inputSize;
        if (layerId == "outputLayer") initialOutputSize = outputSize;
    } else if (layerId in layers) {
        layers[layerId]["layerConfig"] = {
            inputSize,
            outputSize,
            activation,
            type: layers[layerId]["layerConfig"]["type"]
        };
    } else {
        console.log("Layer does not exist!");
        return;
    }

    validateLayers();
    saveLayers();

    // Change Error Box
    const errorBox = document.getElementById("ConfigurationErrorBox");
    const errorBoxHeader = document.getElementById("ConfigurationErrorBoxHeader");
    const errorBoxList = document.getElementById("ConfigurationErrorBoxList");

    if (layers[layerId] && layers[layerId]["valid"]) {
        errorBox.style.display = "none";
    } else {
        errorBox.style.display = "block";
        errorBoxHeader.innerHTML = "Invalid Layer Configuration";
        errorBoxList.innerHTML = "";

        for (const error of layers[layerId]["errors"]) {
            errorBoxList.innerHTML += "<li>" + error +
                "</li>";
        }
    }

    console.log(layers);
}

function loadLayerConfiguration(layerId) {

    // Hide Placeholder
    document.getElementById("layerConfigurationPlaceholder").style.display = "none";
    document.getElementById("layerConfigurationOptionsWindow").hidden = false;

    if (!(layerId in layers) && layerId != "inputLayer" && layerId != "outputLayer") {
        console.log("Layer does not exist!");
        return;
    }

    const isInputLayer = layerId === "inputLayer";
    const isOutputLayer = layerId === "outputLayer";

    // Show respective inputs
    document.getElementById("inputSizeInputContainer").hidden = isOutputLayer; // Input size is hidden for output layer
    document.getElementById("outputSizeInputContainer").hidden = isInputLayer; // Output size is hidden for input layer
    document.getElementById("activationInputContainer").hidden = isInputLayer || isOutputLayer; // Activation is hidden for input and output layers

    // Load Layer Configuration
    if (isInputLayer || isOutputLayer) {

        // Input Layer Title 
        document.getElementById("layerConfigurationTitle").innerHTML = (isInputLayer ? "Input" : "Output") + " Layer Configuration"

        // Configuration Values
        document.getElementById("inputSize").value = initialInputSize;
        document.getElementById("outputSize").value = initialOutputSize;

        // Hide Error Box and Delete Layer Button
        document.getElementById("ConfigurationErrorBox").style.display = "none";
        document.getElementById("deleteLayerConfigurationButton").style.display = "none";

        // Make Input Size Read Only if it is the output layer as it is binary classification
        document.getElementById("outputSize").readOnly = true;

    } else {

        const layer = layers[layerId];

        // Layer Title
        document.getElementById("layerConfigurationTitle").innerHTML = layer.layerName + " Configuration";

        // Configuration Values
        document.getElementById("inputSize").value = layer.layerConfig.inputSize;
        document.getElementById("outputSize").value = layer.layerConfig.outputSize;
        document.getElementById("activation").value = layer.layerConfig.activation;

        // Change Error Box
        document.getElementById("ConfigurationErrorBox").style.display = layer.valid ? "none" : "block";
        document.getElementById("ConfigurationErrorBoxList").innerHTML = "";
        for (const error of layer["errors"]) {
            document.getElementById("ConfigurationErrorBoxList").innerHTML += "<li>" + error + "</li>";
        }

        // Make output size changeable if it is not the last layer
        document.getElementById("outputSize").readOnly = false;
    }

    // Change Button onclick functions
    document.getElementById("saveLayerConfigurationButton").onclick = function () {
        saveLayerConfiguration(layerId);
    };
    document.getElementById("deleteLayerConfigurationButton").onclick = function () {
        deleteLayer(layerId);
    };
}

function deleteLayer(layerId) {

    event.preventDefault();

    // Rename layers and keys to be sequential
    let count = 1;
    const newLayers = {};
    for (const id in layers) {
        const newId = 'layer-' + count;
        if (id != layerId) {
            newLayers[newId] = layers[id];
            count++;
        }
    }

    layers = newLayers;

    // Reload network sidebar
    loadNetwork(layers);

    // Show Placeholder
    document.getElementById("layerConfigurationTitle").innerHTML = "Layer Configuration";
    document.getElementById("layerConfigurationPlaceholder").style.display = "flex";
    document.getElementById("layerConfigurationOptionsWindow").hidden = true;
}

function saveLayers() {
    fetch('/save-layers', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
            'networkId': networkId, 
            'layers': layers, 
            'inputSize': initialInputSize, 
            'outputSize': initialOutputSize 
        }),
    })
        .then((response) => response.json())
        .then((data) => {
            console.log('Success:', data);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
}

function redirectIndex() {
    window.location.href = "/";
}

