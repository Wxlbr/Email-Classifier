function capitalise(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

function generateBlocklistHTML(listName) {
    return `
        <div aria-grabbed="false" id="${listName}"
        class="navMargin flex items-center gap-3 rounded bg-grey-700 padx-3 pady-2 text-grey transition-all hover:text-white cursor-move"
        onclick="selectList('${listName}')">

        <!-- Layer Name -->
        ${capitalise(listName)}

    </div>`;
}

function loadBlocklists() {
    fetch('/get-blocklists', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(response => response.json())
    .then(data => {
        console.log(data);

        const blocklists = data.blocklists;
        const listsView = document.getElementById("listsView");
        listsView.innerHTML = "";

        Object.keys(blocklists).forEach(key => {
            console.log(key);
            listsView.innerHTML += generateBlocklistHTML(key);
        });
    });
}

function addToList(listName, item) {
    togglePopup(listName);

    fetch('/add-blocklist-item', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'listName': listName, 'value': item })
    })
        .then((response) => response.json())
        .then((data) => {
            console.log('Success:', data);

            // Reload List
            selectList(listName);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
}

function listValueHTML(listName, value) {
    return `<a class="button button-grey mr-4 mb-4" id="${value}"
        onclick="removeFromList('${listName}', '${value}')">
        ${value}
    </a>`
    // TODO: Add CSS for button::hover to make pointer
}

function selectList(listName) {
    // Hide Placeholder
    document.getElementById("listConfigurationPlaceholder").style.display = "none";
    document.getElementById("listConfigurationOptionsWindow").hidden = false;

    // Set List Name
    document.getElementById("listConfigurationTitle").innerHTML = capitalise(listName) + " Configuration";

    // Delete previous list values
    document.getElementById("listValues").innerHTML = "";

    // Set List Values
    fetch('/get-blocklists', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(response => response.json())
    .then(data => {
        console.log(data);

        const list = data.blocklists[listName];
        const listsView = document.getElementById("listValues");
        listsView.innerHTML = "";

        console.log(list);

        Object.values(list).forEach(value => {
            console.log(value);
            listsView.innerHTML += listValueHTML(listName, value);
        });
    });

    // Set New Item Popup
    const popup = document.getElementById('newItemPopupToggle');
    popup.setAttribute("onclick", `togglePopup('${listName}')`);
}

function removeFromList(listName, value) {
    console.log(listName, value);

    // event.preventDefault();

    fetch('/remove-blocklist-item', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 'listName': listName, 'value': value })
    })
        .then((response) => response.json())
        .then((data) => {
            console.log('Success:', data);

            // Reload List
            selectList(listName);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
}

function togglePopup(list) {
    console.log(list);

    const popup = document.getElementById('newListItemPopup');
    const isHidden = popup.innerHTML === "";

    console.log(isHidden);

    if (!isHidden) {
        popup.innerHTML = "";
        popup.style.display = "none";
        return;
    }

    popup.style.display = "block";
    popup.innerHTML = `<div class="popup-content card pad-4 flex-col items-center">
        <h1 class="subtitle mb-4">
            New ${capitalise(list)} Item</h1>
        <p id="invalidEmailMsg" class="text-red-500 mb-4 hidden">Invalid Email</p>
        <input class="button border bg-grey-700 text-gray-300 w-full" type="text"
            id="newListItemInput" placeholder="Enter new item" oninput="checkEmailInput('${list}')">
        </input>
        <div class="flex justify-between mt-4">
            <button class="button button-green mr-2" type="submit" id="addNewItemButton" onclick="addToList('${list}', '')" disabled>
                Add
            </button>
            <button class="button button-red bg-red-500" type="submit"
                onclick="togglePopup('${list}')">
                Cancel
            </button>
        </div>
    </div>`;
}

function validEmail(email) {
    const re = /\S+@\S+\.\S+/;
    return re.test(email);
}

function checkEmailInput(list) {
    console.log('Checks Email Input')
    const input = document.getElementById('newListItemInput').value;
    const button = document.getElementById('addNewItemButton');
    const msg = document.getElementById('invalidEmailMsg');

    let valid = true;

    if (input === "" || !validEmail(input)) {
        valid = false;
        msg.innerHTML = "Invalid Email";
    }

    fetch ('/get-blocklists', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
        .then(response => response.json())
        .then(data => {
            console.log(data);

            const blocklists = data.blocklists;
            
            // Check if email already exists in any blocklist
            Object.keys(blocklists).forEach(key => {
                // console.log(key);
                if (blocklists[key].includes(input)) {
                    valid = false;
                    msg.innerHTML = "Email already exists in another blocklist";
                }
            });

            if (valid) {
                button.removeAttribute("disabled");
                msg.classList.add("hidden");
                button.setAttribute("onclick", `addToList('${list}', '${input}')`);
            } else {
                button.setAttribute("disabled", "");
                msg.classList.remove("hidden");
                button.removeAttribute("onclick");
            }

            console.log(valid);
        });
}

function redirectIndex() {
    window.location.href = "/";
}