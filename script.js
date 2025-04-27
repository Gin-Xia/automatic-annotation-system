let selectedImagePath = "";

async function chooseFolder() {
    const folder = await window.pywebview.api.choose_folder();
    if (folder) {
        document.getElementById('folderPathDisplay').innerText = folder;
        await listFiles(folder);
    }
}



async function listFiles(folder) {
    const files = await window.pywebview.api.list_files(folder);
    const list = document.getElementById('fileList');
    list.innerHTML = '';
    files.forEach(file => {
        const li = document.createElement('li');
        li.innerText = file;
        li.onclick = () => previewImage(folder + '/' + file);
        list.appendChild(li);
    });
}

function previewImage(path) {
    const img = document.getElementById('preview');
    const placeholder = document.getElementById('placeholder-text');
    img.src = path;
    img.style.display = 'block';
    placeholder.style.display = 'none';
}


async function process() {
    const prompt = document.getElementById('prompt').value;
    const mode = document.getElementById('mode').value;
    if (selectedImagePath && prompt) {
        const processedPath = await window.pywebview.api.process_image(selectedImagePath, prompt, mode);
        const img = document.getElementById('preview');
        img.src = processedPath;
    } else {
        alert("Please select an image and enter a prompt.");
    }
}



