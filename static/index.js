function toggleActiveNetwork() {
    const activate = activateNetworkButtonText.innerHTML == "On";
    console.log('activate', activate);

    document.getElementById("activeNetworkStatus").innerHTML = activate ? "Status: Active" : "Status: Inactive";
    document.getElementById("activateNetworkButton").classList.toggle("button-green", !activate);
    document.getElementById("activateNetworkButton").classList.toggle("button-red", activate);
    document.getElementById("activateNetworkButtonText").innerHTML = activate ? "Off" : "On"

    fetch('/toggle-active-network', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ activate }),
    })
    .then(response => response.json())
    .catch(error => console.error('Error:', error));
}

function loadNetworks() {

    fetch("/get-networks", {
        method: "GET"
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        loadNetworksHelper(data['active'], data['inactive']);
    })
    .catch(error => {
        console.error("Error fetching networks:", error);
    });
}

function loadNetworksHelper(activeNetwork = {}, inactiveNetworks = {}) {

    const savedNetworks = document.getElementById("savedNetworks");
    const activeNetworkContainer = document.getElementById("activeNetworkContainer");
    const addNetworkCard = document.getElementById("addNetworkCard");

    // Clear saved networks and active network container
    savedNetworks.innerHTML = "";
    activeNetworkContainer.innerHTML = "";

    Object.entries(inactiveNetworks).forEach(([networkId, network]) => {
        savedNetworks.innerHTML += generateInactiveNetworkCardHTML(networkId);
        if (Object.keys(network.network).length !== 0) {
            const networkElement = document.getElementById(networkId);
            ["#trainingStatus", "#trainingAccuracy"].forEach(id => 
                networkElement.querySelector(id).hidden = false
            );
            networkElement.querySelector("#trainingAccuracy").innerHTML = `${network.network.accuracy}%`;
        }
    });

    Object.entries(activeNetwork).forEach(([networkId, network]) => {
        activeNetworkContainer.innerHTML = generateActiveNetworkCardHTML(networkId, network.status);
        if (network.status === 'active') {
            const activeNetworkCard = document.getElementById(networkId);
            ["#activeViewButton", "#activeReplaceButton"].forEach(id => 
                activeNetworkCard.querySelector(id).disabled = true
            );
        }
    });

    // Add new network card
    savedNetworks.innerHTML += addNetworkCard.outerHTML;

    console.log(inactiveNetworks);

    // Only inactive networks need to be validated as active networks are already validated
    validateNetworks(inactiveNetworks);

    // Check for any networks currently being trained
    fetch("/current-training", {
        method: "GET"
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        data.ids.forEach(id => showTrainingCard(id));
    })
    .catch(error => {
        console.error("Error fetching networks:", error);
    });
}

function loadBlocklists() {
    fetch("/get-blocklists", {
        method: "GET"
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        loadBlocklistsHelper(data);
    })
    .catch(error => {
        console.error("Error fetching blocklists:", error);
    });
}

function loadBlocklistsHelper(blocklists) {
    const blocklistContainer = document.getElementById("blocklistContainer");
    blocklistContainer.innerHTML = generateBlocklistCardHTML(blocklists);
}

function generateBlocklistCardHTML(data) {
    let output = '';

    let blocklists = data['blocklists']

    for (let key in blocklists) {
        if (blocklists.hasOwnProperty(key)) {
            output += `${key}: ${blocklists[key].length}<br>`;
        }
    }

    return `<div class="card" id="blocklistsCard">
    <div class="h-65 bg-grey-600 pad-4 rounded-top">
        <h2 class="card-title">Blocklist Elements</h2>
        <span id="blocklistStatus">${output}</span>
    </div>
    <div class="h-35 pad-4 items-center rounded-bottom">
        <button class="button button-blue" type="submit" onclick="redirectEditBlocklists()">Edit</button>
    </div>
    </div>`;
}

function redirectEditBlocklists() {
    if (socket != null) {
        socket.disconnect();
    }

    window.location.href = "/edit-blocklists";
}

function generateInactiveNetworkCardHTML(networkId) {
    return `<div class="card" id = "${networkId}">
    <div class="h-65 bg-grey-600 pad-4 rounded-top">
        <div style="display: flex; align-items: center;">
            <h2 class="card-title">Network ${networkId.split('-')[1]}</h2>
            <!-- Network Status -->
            <span class="ml-2 bg-red-500 text-dark-grey padx-2 pady-1 rounded-full text-xs" id="validationStatus">Invalid</span>
            <span class="ml-2 bg-blue-500 text-dark-grey padx-2 pady-1 rounded-full text-xs" id="trainingStatus" hidden>Trained</span>
            <span class="ml-2 bg-blue-500 text-dark-grey padx-2 pady-1 rounded-full text-xs" id="trainingAccuracy" hidden></span>
        </div>
        <!-- Training Progress -->
        <div id="trainingCard" hidden>
            <div class="ml-1">
                <h3 class="text-sm text-semi-bold">Training Progress</h3>
                <span class="text-sm" id="epoch">Preparing Environment...</span>
                <span class="text-sm" id="error"></span>
                <span class="text-sm" id="accuracy"></span>
                <span class="text-sm" id="eta"></span>
            </div>

            <div class="mt-1" style="display: flex; align-items: center;">
                <progress class="progress-bar" value="0" max="100" id="trainingProgressBar"></progress>
                <span class="text-sm ml-2 mt-375" id="trainingProgressPercentage">0%</span>
            </div>
        </div>
    </div>
    <div class="h-35 pad-4 items-center rounded-bottom">
        <button class="button button-blue" type="submit" onclick="selectActiveNetwork('${networkId}')" style="display: none;"
            id="selectButton">
            Select
        </button>
        <button onclick="redirectEditNetwork('${networkId}')"
            class="button button-blue" type="submit" id="editButton">
            Edit
        </button>
        <button onclick="toggleTrainPopup('${networkId}')"
            class="button button-blue ml-4" type="submit" id="trainButton">
            Train
        </button>
        <button onclick="deleteNetwork('${networkId}')" 
        class="button button-red ml-4" type="submit" id="deleteButton">
            Delete
        </button>
        <button onclick="stopTraining('${networkId}')" 
        class="button button-red" type="submit" id="stopTrainingButton" style="display: none;">
            Stop Training
        </button>
    </div>
    </div>`;
}

function generateActiveNetworkCardHTML(networkId, status) {
    return `<div class="card" id="${networkId}">
    <div class="h-65 bg-grey-600 pad-4 rounded-top">
        <h2 class="card-title">Network ${networkId.split('-')[1]}</h2>
        <span id="activeNetworkStatus">Status: ${status === 'active' ? 'Active' : 'Inactive'}</span >
    </div >

    <div class="h-35 pad-4 items-center rounded-bottom">
            <!-- Disable (grey) whilst network is active -->
        <button class="button button-blue" type="submit" onclick="selectActiveNetwork('${networkId}')" style="display: none;"
            id="activeSelectButton">Select</button>
        <button class="button button-${status === 'active' ? 'red' : 'green'} border" id="activateNetworkButton"
            onclick="toggleActiveNetwork()">
            <svg class="w-6 h-6" xmlns="http://www.w3.org/2000/svg" width="24" height="24"
                viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                stroke-linecap="round" stroke-linejoin="round">
                <path d="M18.36 6.64a9 9 0 1 1-12.73 0"></path>
                <line x1="12" x2="12" y1="2" y2="12"></line>
            </svg>
            <p class="ml-1" id="activateNetworkButtonText">${status === 'active' ? 'Off' : 'On'}</p>
        </button>
        <button class="button button-blue ml-4" type="submit" onclick="replaceActiveNetwork()"
            id="activeReplaceButton">Replace</button>
    </div>
    </div>`;
}

function addNetwork() {

    fetch('/save-new-network', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then((response) => response.json())
    .then((data) => {
        console.log(data);
        loadNetworks();
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function redirectEditNetwork(networkId) {
    if (!document.getElementById(networkId).querySelector("#trainingStatus").hidden) {
        if (!confirm('Network is trained, editing will reset training. Are you sure you want to continue?')) {
            return;
        }
    }

    if (socket != null) {
        socket.disconnect();
    }

    window.location.href = `/edit-network?networkId=${networkId}`;
}

function validateNetworks(networksToValidate) {

    Object.entries(networksToValidate).forEach(([networkId, { valid }]) => {

        const validationStatus = document.querySelector(`#${networkId} #validationStatus`);

        validationStatus.innerHTML = valid ? "Valid" : "Invalid";
        validationStatus.classList.toggle("bg-green-500", valid);
        validationStatus.classList.toggle("bg-red-500", !valid);

        if (!valid) {
            fetch("/remove-trained", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ "networkId": networkId })
            })
            .then(response => response.json())
            .then( console.log )
            .catch(error => {
                console.error("Error removing trained:", error);
            });
        }
    })
}

function loadTrainingStatus(networkId, data) {
    // Get network card
    let trainingCard = document.getElementById(networkId).querySelector("#trainingCard");
    trainingCard.hidden = false;

    // Set training progress
    trainingCard.querySelector("#trainingProgressBar").value = (data.epoch - 1) / data.epochs * 100;
    trainingCard.querySelector("#trainingProgressPercentage").innerHTML = `${Math.round((data.epoch - 1) / data.epochs * 100)}% `;

    // Set epoch, error, accuracy and eta
    trainingCard.querySelector("#epoch").innerHTML = `Epoch ${data.epoch}/${data.epochs},`;
    trainingCard.querySelector("#error").innerHTML = `Error: ${data.error},`;
    trainingCard.querySelector("#accuracy").innerHTML = `Accuracy: ${data.accuracy}%,`;
    trainingCard.querySelector("#eta").innerHTML = `ETA: ${data.totalEta}`;

    // TODO: Check buttons are disabled 
    let networkCard = document.getElementById(networkId);

    // Hide training card
    networkCard.querySelector("#trainingCard").hidden = false;

    networkCard.querySelector("#editButton").style.display = "none";
    networkCard.querySelector("#trainButton").style.display = "none";
    networkCard.querySelector("#deleteButton").style.display = "none";
}

function clearTrainingStatus(networkId) {
    // Get network card
    let trainingCard = document.getElementById(networkId).querySelector("#trainingCard");
    trainingCard.hidden = true;

    // Set training progress
    trainingCard.querySelector("#trainingProgressBar").value = 0;
    trainingCard.querySelector("#trainingProgressPercentage").innerHTML = `0% `;

    // Set epoch, error, accuracy and eta
    trainingCard.querySelector("#epoch").innerHTML = `Preparing Environment...`;
    trainingCard.querySelector("#error").innerHTML = ``;
    trainingCard.querySelector("#accuracy").innerHTML = ``;
    trainingCard.querySelector("#eta").innerHTML = ``;
}

function initTrainingSocket() {
    const socket = io.connect(`http://${window.location.hostname}:${location.port}/train`);

    // Listeners for socket connect and disconnect
    ['connect', 'disconnect'].forEach(event => 
        socket.on(event, () => console.log(`Socket ${event}ed`))
    );

    // Listener for training updates
    socket.on('training_update', data => {
        console.log('Received training update:', data);
        if (document.getElementById(data.networkId)) {
            loadTrainingStatus(data.networkId, data);
        }
    });

    // Listener for training done
    socket.on('training_done', data => {
        console.log('Received training done:', data);

        let networkCard = document.getElementById(data.networkId);
        let trainingCard = networkCard.querySelector("#trainingCard");

        trainingCard.hidden = true;

        // Hide training card and show other buttons
        ["#editButton", "#trainButton", "#deleteButton"].forEach(id => 
            networkCard.querySelector(id).style.display = "block"
        );

        networkCard.querySelector("#stopTrainingButton").style.display = "none";

        // Set network to trained
        let isCancelled = data.cancelled;
        ["#trainingStatus", "#trainingAccuracy"].forEach(id => 
            networkCard.querySelector(id).hidden = isCancelled
        );

        // Set network accuracy
        networkCard.querySelector("#trainingAccuracy").innerHTML = `${data.accuracy}%`;

        // Clear training status
        clearTrainingStatus(data.networkId);
    });

    return socket;
}

function trainNetwork(networkId, epochs) {

    // Get network
    fetch ("/get-network", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ "networkId": networkId })
    })  
    .then(response => response.json())
    .then(data => {
        console.log(data);
        trainNetworkHelper(data, epochs);
    })
    .catch(error => {
        console.error("Error fetching network:", error);
    });
}

function showTrainingCard(networkId) {

    // Get network card
    let networkCard = document.getElementById(networkId);

    // Show training card
    networkCard.querySelector("#trainingCard").hidden = false;
    networkCard.querySelector("#editButton").style.display = "none";
    networkCard.querySelector("#trainButton").style.display = "none";
    networkCard.querySelector("#deleteButton").style.display = "none";
    networkCard.querySelector("#stopTrainingButton").style.display = "block";
}

function trainNetworkHelper(network, epochs) {

    // Get network
    const networkId = network.networkId;

    console.log('Training network:', networkId);

    // Get network card
    let networkCard = document.getElementById(networkId);
    let valid = networkCard.querySelector("#validationStatus").innerHTML == "Valid";

    if (Object.keys(network.network).length !== 0) {
        if (!confirm(`Network is trained, continuing will reset training. 
            Are you sure you want to continue?`)) {
            return;
        }
    
        document.getElementById(networkId).querySelector("#trainingStatus").hidden = true;
        document.getElementById(networkId).querySelector("#trainingAccuracy").hidden = true;

        console.log('Ok, redirecting to edit network');
    }

    if (!valid) {
        return alert("Network is invalid, please edit and validate before training");
    }

    showTrainingCard(networkId);

    fetch("/train-network", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ "networkId": networkId, "epochs": epochs })
    })
    .then(response => response.json())
    .then(console.log)
    .catch(error => console.error("Error training network:", error) );
}

function stopTraining(networkId) {
    fetch("/stop-training", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ "networkId": networkId })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);

        // Get network card
        let networkCard = document.getElementById(networkId);

        // Hide training card
        networkCard.querySelector("#trainingCard").hidden = true;

        // Set edit, train and delete buttons to enabled
        networkCard.querySelector("#editButton").style.display = "block";
        networkCard.querySelector("#trainButton").style.display = "block";
        networkCard.querySelector("#deleteButton").style.display = "block";
        networkCard.querySelector("#stopTrainingButton").style.display = "none";
    })
    .catch(error => {
        console.error("Error stopping training:", error);
    });
}

function deleteNetwork(networkId) {
    if (!confirm('Are you sure you want to delete this network?')) {
        return;
    }

    fetch("/delete-network", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ "networkId": networkId })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        loadNetworks();
    })
    .catch(error => {
        console.error("Error deleting network:", error);
    });
}

// TODO: Hide Stop Training button in selection view

function replaceActiveNetwork() {
    let activeNetworkStatus = document.getElementById("activeNetworkStatus");

    if (activeNetworkStatus.innerHTML == "Status: Active") {
        return alert("Active network must be deactivated before replacing");
    }

    if (!confirm('Are you sure you want to replace the active network?')) {
        return console.log('Not replacing network');
    }

    let activeNetworkCard = document.querySelector("#activeNetworkContainer .card");
    let savedNetworkCards = document.querySelectorAll("#savedNetworks .card");

    ["#activeViewButton", "#activeReplaceButton", "#activateNetworkButton"].forEach(id => 
        activeNetworkCard.querySelector(id).style.display = "none"
    );

    activeNetworkCard.querySelector("#activeNetworkStatus").innerHTML = "";
    activeNetworkCard.querySelector("#activeSelectButton").style.display = "block";

    savedNetworkCards.forEach((savedNetworkCard, i) => {
        if (i < savedNetworkCards.length - 1) {
            ["#editButton", "#trainButton", "#deleteButton"].forEach(id => 
                savedNetworkCard.querySelector(id).style.display = "none"
            );

            savedNetworkCard.querySelector("#selectButton").style.display = "block";

            if (savedNetworkCard.querySelector("#trainingStatus").hidden) {
                savedNetworkCard.style.opacity = 0.5;
                savedNetworkCard.querySelector("#selectButton").disabled = true;
            }
        }
    });

    let addNetworkCard = document.getElementById("addNetworkCard");
    addNetworkCard.style.opacity = 0.5;
    addNetworkCard.querySelector("#addNetworkButton").disabled = true;
}

function selectActiveNetwork(newNetworkId) {

    console.log('selectActiveNetwork', newNetworkId)

    fetch('/switch-active-network', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'newNetworkId': newNetworkId }),
    })
    .then((response) => response.json())
    .then((data) => {
        console.log('Success:', data);
        loadNetworks();
    })
    .catch((error) => {
        console.error('Error:', error);
    });

    // Activate add network card
    let addNetworkCard = document.getElementById("addNetworkCard");
    addNetworkCard.style.opacity = 1;
    addNetworkCard.querySelector("#addNetworkButton").disabled = false;
}

function toggleTrainPopup(networkId) {

    // Get network card
    let networkCard = document.getElementById(networkId);
    let valid = networkCard.querySelector("#validationStatus").innerHTML == "Valid";

    // If network is not valid, then cannot train
    if (!valid) {
        return alert("Network is invalid, please edit and validate before training");
    }

    const popup = document.getElementById('trainNetworkPopup');
    if (popup.style.display === "none") {
        // Reset the popup
        popup.style.display = "block";

        // Set button parameters
        popup.querySelector("#trainNetworkButton").onclick = function () { trainNetworkFromPopUp(networkId) };
        popup.querySelector("#trainNetworkCancelButton").onclick = function () { toggleTrainPopup(networkId) };

        popup.querySelector("#trainNetworkPopupTitle").innerHTML = `Train Network ${networkId.split('-')[1]}`;
    } else {
        popup.style.display = "none";
    }
}

function trainNetworkFromPopUp(networkId) {

    console.log('trainNetworkFromPopUp', networkId);

    const epochs = parseInt(document.getElementById('trainNetworkEpochs').value);

    toggleTrainPopup(networkId);

    if (epochs > 0 && epochs <= 1000) {
        trainNetwork(networkId, epochs);
    } else {
        alert("Number of epochs must be between 1 and 1000");
    }
}