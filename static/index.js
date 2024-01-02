// function writeToCache(key, value) {
//     localStorage.setItem(key, JSON.stringify(value));
// }

// function readFromCache(key) {
//     const cachedData = localStorage.getItem(key);
//     return cachedData ? JSON.parse(cachedData) : null;
// }

function toggleActiveLayer() {

    const activate = activateNetworkLabel.innerHTML == "Activate" ? true : false;

    console.log('activate', activate);

    document.getElementById("activeNetworkStatus").innerHTML = activate ? "Status: Active" : "Status: Inactive";
    document.getElementById("activateNetworkLabel").innerHTML = activate ? "Deactivate" : "Activate";
    document.getElementById("activateNetworkButton").classList.toggle("button-green", activate);
    document.getElementById("activateNetworkButton").classList.toggle("button-red", !activate);
    document.getElementById("activeViewButton").disabled = activate ? true : false;
    document.getElementById("activeReplaceButton").disabled = activate ? true : false;

    fetch('/toggle-active-network', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'activate': activate }),
    })
    .then((response) => response.json())
    .then(() => {})
    .catch((error) => {
        console.error('Error:', error);
    });
}

function getNetworks(func) {

    fetch("/get-networks", {
        method: "GET",
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        func(data);
    })
    .catch(error => {
        console.error("Error fetching networks:", error);
    });
}

function loadNetworks(networksToLoad = {}) {

    if (Object.keys(networksToLoad).length == 0) {
        console.log("getNetworks");
        getNetworks(loadNetworks);
        return;
    }

    let savedNetworks = document.getElementById("savedNetworks");
    let activeNetworkContainer = document.getElementById("activeNetworkContainer");
    let addNetworkCard = document.getElementById("addNetworkCard");

    // Clear saved networks and active network container
    savedNetworks.innerHTML = "";
    activeNetworkContainer.innerHTML = "";

    for (networkId in networksToLoad) {
        const activeCard = networksToLoad[networkId].activeCard;

        if (activeCard) {

            console.log("activeCard", networkId);

            let status = networksToLoad[networkId].status;

            activeNetworkContainer.innerHTML = `<div class="card" id="${networkId}">
            <div class="h-65 bg-grey-600 pad-4 rounded-top">
                <h2 class="card-title">Network ${networkId.split('-')[1]}</h2>
                <span id="activeNetworkStatus">Status: ${status === 'active' ? 'Active' : 'Inactive'}</span >
            </div >

            <div class="h-35 pad-4 items-center rounded-bottom">
                    <!-- Disable (grey) whilst network is active -->
                <button class="button button-blue" type="submit" onclick="selectActiveNetwork('${networkId}')" style="display: none;"
                    id="activeSelectButton">Select</button>
                <button class="button button-blue" type="submit" onclick=""
                    id="activeViewButton">View</button>
                <button class="button button-blue ml-4" type="submit" onclick="replaceActiveNetwork()"
                    id="activeReplaceButton">Replace</button>
                <button class="button button-square button-${status === 'active' ? 'red' : 'green'} border ml-4" id="activateNetworkButton"
                    onclick="toggleActiveLayer()">
                    <svg class="w-6 h-6" xmlns="http://www.w3.org/2000/svg" width="24" height="24"
                        viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                        stroke-linecap="round" stroke-linejoin="round">
                        <path d="M18.36 6.64a9 9 0 1 1-12.73 0"></path>
                        <line x1="12" x2="12" y1="2" y2="12"></line>
                    </svg>
                </button>
                <span class="ml-4" id="activateNetworkLabel">${status === 'active' ? 'Deactive' : 'Activate'}</span>
            </div>
            </div>`;

            let activeNetworkCard = document.getElementById(networkId);

            if (status === 'active') {
                activeNetworkCard.querySelector("#activeViewButton").disabled = true;
                activeNetworkCard.querySelector("#activeReplaceButton").disabled = true;
            }

            continue;
        }

        savedNetworks.innerHTML += `<div class="card" id = "${networkId}">
        <div class="h-65 bg-grey-600 pad-4 rounded-top">
            <div style="display: flex; align-items: center;">
                <h2 class="card-title">Network ${networkId.split('-')[1]}</h2>
                <!-- Network Status -->
                <span class="ml-2 bg-red-500 text-dark-grey padx-2 pady-1 rounded-full text-xs" id="validationStatus">Invalid</span>
                <span class="ml-2 bg-blue-500 text-dark-grey padx-2 pady-1 rounded-full text-xs" id="trainingStatus" hidden>Trained</span>
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
            <button onclick="trainNetwork('${networkId}')"
                class="button button-blue ml-4" type="submit" id="trainButton">
                Train
            </button>
            <button onclick="deleteNetwork('${networkId}')" 
            class="button button-red ml-4" type="submit" id="deleteButton">
                Delete
            </button>
        </div>
        </div>`;

        if (Object.keys(networksToLoad[networkId].network).length !== 0) {
            document.getElementById(networkId).querySelector("#trainingStatus").hidden = false;
        }
    }

    // Add new network card
    savedNetworks.innerHTML += addNetworkCard.outerHTML;

    networks = networksToLoad;

    validateNetworks(networksToLoad);
}

function addNetwork() {
    // NetworkId
    let IdNum = Math.floor(999 + Math.random() * 1000);
    while (`network-${IdNum}` in networks) {
        IdNum = Math.floor(999 + Math.random() * 1000);
    }

    console.log(IdNum);

    networks[`network-${IdNum}`] = {
        "networkId": `network-${IdNum}`,
        "name": `Network ${IdNum}`,
        "activeCard": false,
        "layers": {},
        "inputSize": 0,
        "outputSize": 0,
        "valid": false,
        "status": "inactive",
        "network": {}
    };

    saveNetwork(`network-${IdNum}`);

    loadNetworks(networks);
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
    for (let networkId in networksToValidate) {
        const network = networksToValidate[networkId];
        console.log(networkId, network);
        if (network.activeCard) {
            // Skip active card as it has been validated already
            continue;
        }

        const valid = network.valid;
        let validationStatus = document.getElementById(networkId).querySelector("#validationStatus");

        validationStatus.innerHTML = valid ? "Valid" : "Invalid";
        validationStatus.classList.toggle("bg-green-500", valid);
        validationStatus.classList.toggle("bg-red-500", !valid);
    }
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
    trainingCard.querySelector("#eta").innerHTML = `ETA: ${data.epochEta} (${data.totalEta})`;
}

function initTrainingSocket() {

    let namespace = '/train';
    // let room = 'test';

    const socket = io.connect('http://' + document.domain + ':' + location.port + namespace);

    // Add event listeners or any other socket-related logic here
    socket.on('connect', () => {
        console.log('Socket connected');
    });

    socket.on('disconnect', () => {
        console.log('Socket disconnected');
    });

    // Add a listener for training updates
    socket.on('training_update', function(data) {
        console.log('Received training');
        console.log('Received training update:', data);

        // Cache data for page refresh
        // writeToCache(data.networkId, data);

        loadTrainingStatus(data.networkId, data);
    });

    // Add a listener for training done
    socket.on('training_done', function(data) {
        console.log('Received training done');
        console.log('Received training done:', data);

        let networkCard = document.getElementById(data.networkId);
        let trainingCard = networkCard.querySelector("#trainingCard");

        trainingCard.hidden = true;

        // Set edit, train and delete buttons to enabled
        networkCard.querySelector("#editButton").disabled = false;
        networkCard.querySelector("#trainButton").disabled = false;
        networkCard.querySelector("#deleteButton").disabled = false;

        // Set network to trained
        // console.log("setTrained", data.networkId);

        networkCard.querySelector("#trainingStatus").hidden = false;
    });

    return socket;
}

function trainNetwork(networkId) {

    // Get network
    let network = networks[networkId];

    // Get network card
    let networkCard = document.getElementById(networkId);

    let valid = networkCard.querySelector("#validationStatus").innerHTML == "Valid" ? true : false;

    if (Object.keys(network.network).length !== 0) {
        if (!confirm('Network is trained, continuing will reset training. Are you sure you want to continue?')) {
            return;
        }
    
        document.getElementById(networkId).querySelector("#trainingStatus").hidden = true;

        console.log('Ok, redirecting to edit network');
    }

    if (!valid) {
        alert("Network is invalid, please edit and validate before training");
        return;
    }

    // Hide training card
    networkCard.querySelector("#trainingCard").hidden = false;

    // Set edit, train and delete buttons to disabled
    networkCard.querySelector("#editButton").disabled = true;
    networkCard.querySelector("#trainButton").disabled = true;
    networkCard.querySelector("#deleteButton").disabled = true;

    fetch("/train-network", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            "networkId": networkId,
            "network": network
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
    })
    .catch(error => {
        console.error("Error training network:", error);
    });
}

// function checkActiveTraining() {
//     // Check if active network is training
//     for (networkId in networks) {
//         const network = networks[networkId];

//         if (!network.activeCard) {
            
//             // Read from cache
//             let cachedData = readFromCache(networkId);

//             if (cachedData != null) {
//                 loadTrainingSSE(networkId, cachedData);
//             }
//         }
//     }
// }

function deleteNetwork(networkId) {
    if (!confirm('Are you sure you want to delete this network?')) {
        return;
    }

    fetch("/delete-network", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            "networkId": networkId
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        loadNetworks(data);
    })
    .catch(error => {
        console.error("Error deleting network:", error);
    });
}

function replaceActiveNetwork() {

    let activeNetworkStatus = document.getElementById("activeNetworkStatus");

    if (activeNetworkStatus.innerHTML == "Status: Active") {
        alert("Active network must be deactivated before replacing");
    }

    if (!confirm('Are you sure you want to replace the active network?')) {
        console.log('Not replacing network');
        return;
    }

    // Switch to selection view
    // Get active network card
    let activeNetworkCard = document.getElementById("activeNetworkContainer").querySelector(".card");
    
    // Set style to display: none
    activeNetworkCard.querySelector("#activeViewButton").style.display = "none";
    activeNetworkCard.querySelector("#activeReplaceButton").style.display = "none";
    activeNetworkCard.querySelector("#activateNetworkButton").style.display = "none";
    activeNetworkCard.querySelector("#activateNetworkLabel").style.display = "none";
    activeNetworkCard.querySelector("#activeNetworkStatus").innerHTML = "";

    // Show select button on card
    activeNetworkCard.querySelector("#activeSelectButton").style.display = "block";

    // Get all saved network cards
    let savedNetworkCards = document.getElementById("savedNetworks").querySelectorAll(".card");

    // Loop through saved network cards
    // Skip last card as it is the add network card
    for (i = 0; i < savedNetworkCards.length - 1; i++) {
        let savedNetworkCard = savedNetworkCards[i];

        // Set style to display: none
        savedNetworkCard.querySelector("#editButton").style.display = "none";
        savedNetworkCard.querySelector("#trainButton").style.display = "none";
        savedNetworkCard.querySelector("#deleteButton").style.display = "none";

        // Show select button on card
        savedNetworkCard.querySelector("#selectButton").style.display = "block";

        // Hide invalid networks as they cannot be selected
        let trainingStatus = savedNetworkCard.querySelector("#trainingStatus");

        // Check if training status is hidden (Covers both invalid and untrained networks)
        if (trainingStatus.hidden) {

            // Set opacity to 0.5 and disable select button to 'hide' card
            savedNetworkCard.style.opacity = 0.5;
            selectButton = savedNetworkCard.querySelector("#selectButton");
            selectButton.disabled = true;
        }
    }

    // Disable add network card
    let addNetworkCard = document.getElementById("addNetworkCard");
    addNetworkCard.style.opacity = 0.5;
    addNetworkCard.querySelector("#addNetworkButton").disabled = true;
}

function selectActiveNetwork(newNetworkId) {

    console.log('selectActiveNetwork', newNetworkId)

    // TODO: Check if network is not null
    let activeNetworkCard = document.getElementById("activeNetworkContainer").querySelector(".card");   
    let activeNetworkId = activeNetworkCard.id;

    fetch('/switch-active-network', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'newNetworkId': newNetworkId, 'activeNetworkId': activeNetworkId }),
    })
    .then((response) => response.json())
    .then((data) => {
        console.log('Success:', data);
        loadNetworks();
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function saveNetwork(networkId) {
    fetch('/save-network', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'networkId': networkId }),
    })
    .then((response) => response.json())
    .then(() => {})
    .catch((error) => {
        console.error('Error:', error);
    });
}