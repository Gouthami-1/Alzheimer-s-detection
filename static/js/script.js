const fileInput = document.getElementById("mri_image");
const previewBox = document.getElementById("preview-box");
const previewImage = document.getElementById("preview-image");
const submitBtn = document.getElementById("submit-btn");
const btnText = document.getElementById("btn-text");
const btnSpinner = document.getElementById("btn-spinner");
const fileError = document.getElementById("file-error");
const form = document.getElementById("upload-form");

const allowedTypes = ["image/jpeg", "image/png"];
const maxSizeBytes = 5 * 1024 * 1024;

fileInput.addEventListener("change", function () {
    const file = fileInput.files[0];
    fileError.classList.add("d-none");
    submitBtn.disabled = true;
    previewBox.classList.add("d-none");

    if (!file) return;

    if (!allowedTypes.includes(file.type)) {
        fileError.textContent = "Please select a JPG, JPEG or PNG image.";
        fileError.classList.remove("d-none");
        fileInput.value = "";
        return;
    }
    if (file.size > maxSizeBytes) {
        fileError.textContent = "File is larger than 5 MB.";
        fileError.classList.remove("d-none");
        fileInput.value = "";
        return;
    }

    previewImage.src = URL.createObjectURL(file);
    previewBox.classList.remove("d-none");
    submitBtn.disabled = false;
});

form.addEventListener("submit", function () {
    submitBtn.disabled = true;
    btnText.textContent = "Analyzing...";
    btnSpinner.classList.remove("d-none");
});