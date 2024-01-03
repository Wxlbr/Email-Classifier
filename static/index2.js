function redirectIndex() {
    window.location.href = "/";
}

function loadNetwork() {
    fetch('/get-layers', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'networkId': networkId }),
    })
    .then((response) => response.json())
    .then((data) => {
        console.log('Success:', data);
        loadNetworkHelper(data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function loadNetworkHelper(layers) {
    // Clear network sidebar
    document.getElementById("layersView").innerHTML = "";

    let layersView = document.getElementById("layersView");

    for (const layerId in layers) {
        layersView.innerHTML += `<div aria-grabbed="false" id="${layerId}"
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
        ${layers[layerId]["layerName"]}

        <!-- Layer Status -->
        <span class="ml-auto bg-green-500 text-dark-grey padx-2 pady-1 rounded-full text-xs"
            id="validationStatus">Valid</span>
        </div>`;
    }

    saveLayers();
}

function addLayer(layerName, layerType) {
    fetch('/add-layer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'layerName': layerName, 'layerType': layerType }),
    })
    .then((response) => response.json())
    .then((data) => {
        console.log('Success:', data);
        loadNetwork();
    })
    .catch((error) => {
        console.error('Error:', error);
    });
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

function updateLayerOrder(event) {
    fetch ('/update-layer-order', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'networkId': networkId, 'oldIndex': event.oldIndex, 'newIndex': event.newIndex }),
    })  
    .then((response) => response.json())
    .then((data) => {
        console.log('Success:', data);
        loadNetwork();
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function togglePopup(elementId) {
    const popup = document.getElementById(elementId);
    if (popup.style.display === "none") {
        // Reset the popup
        popup.style.display = "block";

        // Disable the add button
        document.getElementById("newLayerPopupAddButton").disabled = true;
    } else {
        popup.style.display = "none";

        // Remove any selected buttons
        for (const button of document.getElementsByClassName("layer-button")) {
            button.classList.remove("selected");
        }
    }
}

function selectLayerType(layerName, layerType) {
    // Change the colour of the selected button
    const layerButton = document.getElementById("newLayerSelection-" + layerType);
    layerButton.classList.add("selected");

    // Change the colour of the other buttons
    for (const button of document.getElementsByClassName("layer-button")) {
        if (button.id != layerButton.id) {
            button.classList.remove("selected");
        }
    }

    // Change the onclick function of the add button to add the selected layer type
    const addButton = document.getElementById("newLayerPopupAddButton");
    addButton.onclick = function () {
        // Close and reset the popup
        togglePopup("newLayerPopup");

        // Add the layer
        addLayer(layerName, layerType);
    };

    // Activate the add button
    addButton.disabled = false;
}

function saveLayerConfigurationHelper(layerId, layers) {

    if (layerId == "inputLayer") {
        inputSize = document.getElementById("inputSize").value;
        saveLayers();
        return;
    }

    if (layerId == "outputLayer") {
        outputSize = document.getElementById("outputSize").value;
        saveLayers();
        return;
    }

    if (!(layerId in layers)) {
        console.log("Layer does not exist!");
        return;
    }

    // Get Layer Configuration Values
    layers[layerId]["layerConfig"] = {
        "inputSize": document.getElementById("inputSize").value,
        "outputSize": document.getElementById("outputSize").value,
        "activation": document.getElementById("activation").value,
        "type": layers[layerId]["layerConfig"]["type"]
    };

    saveLayers();

    // Change Error Box
    if (layers[layerId]["valid"]) {
        document.getElementById("ConfigurationErrorBox").style.display = "none";
    } else {
        document.getElementById("ConfigurationErrorBox").style.display = "block";
        document.getElementById("ConfigurationErrorBoxHeader").innerHTML = "Invalid Layer Configuration";
        document.getElementById("ConfigurationErrorBoxList").innerHTML = "";

        for (const error of layers[layerId]["errors"]) {
            document.getElementById("ConfigurationErrorBoxList").innerHTML += "<li>" + error +
                "</li>";
        }
    }

    console.log(layers);
}

function saveLayerConfiguration(layerId) {
    
    console.log("Saving layer configuration for layer: " + layerId + "..." + networkId);

    return;

    fetch('/get-layers', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'networkId': networkId }),
    })
    .then((response) => response.json())
    .then((data) => {
        console.log('Success:', data);
        saveLayerConfigurationHelper(layerId, data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function loadLayerConfiguration(layerId) {
    fetch('/get-layers', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'networkId': networkId }),
    })
    .then((response) => response.json())
    .then((data) => {
        console.log('Success:', data);
        loadLayerConfigurationHelper(layerId, data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function loadLayerConfigurationHelper(layerId, layers) {

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
        document.getElementById("inputSize").value = (isInputLayer ? inputSize : outputSize);

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

    fetch('/delete-layer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'networkId': networkId, 'layerId': layerId }),
    })
    .then((response) => response.json())
    .then((data) => {
        console.log('Success:', data);

        // Reload network sidebar
        loadNetwork();

        // Show Placeholder
        document.getElementById("layerConfigurationTitle").innerHTML = "Layer Configuration";
        document.getElementById("layerConfigurationPlaceholder").style.display = "flex";
        document.getElementById("layerConfigurationOptionsWindow").hidden = true;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function saveLayers() {
    fetch ('/validate-layers', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'networkId': networkId }),
    })
    .then((response) => response.json())
    .then((data) => {
        console.log('Success:', data);

        const layers = data["layers"];

        for (const layerId in layers) {
            // Update validation status
            let validationStatus = document.getElementById(layerId).querySelector("#validationStatus");
            const valid = layers[layerId]["valid"];

            validationStatus.innerHTML = valid ? "Valid" : "Invalid";
            validationStatus.classList.toggle("bg-green-500", valid);
            validationStatus.classList.toggle("bg-red-500", !valid);
        }
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}