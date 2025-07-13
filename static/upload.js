document.addEventListener('DOMContentLoaded', () => {
  let animeData = {};            // filled in elsewhere via parse_fight
  let uploadedFiles = [];

  const form         = document.getElementById('uploadForm');
  const uploadStatus = document.getElementById('uploadStatus');
  const compileBtn   = document.getElementById('compile-btn');
  const resultDiv    = document.getElementById('result');

  // ----------------------------
  // Upload handler
  // ----------------------------
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

console.log(
  document.getElementById('file-input'),
  document.getElementById('upload-btn'),
  document.getElementById('uploadStatus'),
  document.getElementById('compile-btn'),
  document.getElementById('result')
);



    // UI feedback
    uploadStatus.textContent = 'Uploading...';
    compileBtn.disabled = true;
    resultDiv.innerHTML = '';

    const data = new FormData(form);

    try {
      const res  = await fetch(form.action, { method: 'POST', body: data });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Upload failed');

      // Expecting { saved_files: [...] }
      uploadedFiles = json.saved_files || [json.filename];
      uploadStatus.textContent = `✅ Uploaded ${uploadedFiles.length} file(s).`;
      compileBtn.disabled = false;
    } catch (err) {
      console.error(err);
      uploadStatus.textContent = `❌ ${err.message}`;
      alert('Upload error: ' + err.message);
    }
  });

  // ----------------------------
  // Compile handler
  // ----------------------------
  compileBtn.addEventListener('click', async () => {
    if (!uploadedFiles.length) {
      return alert('Please upload episodes first.');
    }

    compileBtn.disabled = true;
    resultDiv.innerHTML = 'Compiling…';

    try {
      // 1) Get timestamps
      const tsRes = await fetch('/get_timestamps', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(animeData)
      });
      if (!tsRes.ok) throw new Error(`Timestamps error: ${tsRes.status}`);
      const { timestamps } = await tsRes.json();

      // 2) Compile fight
      const compileRes = await fetch('/compile_fight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          timestamps,
          video_files: uploadedFiles
        })
      });
      if (!compileRes.ok) throw new Error(`Compile error: ${compileRes.status}`);
      const { output_file } = await compileRes.json();

      // 3) Show result
      const videoUrl = `/output/${output_file}`;
      resultDiv.innerHTML = `
        <h3>✅ Your fight is ready:</h3>
        <video controls width="600" src="${videoUrl}"></video><br>
        <a href="${videoUrl}" download>Download Video</a>
      `;
    } catch (err) {
      console.error(err);
      resultDiv.innerHTML = '';
      alert('Error compiling fight: ' + err.message);
    } finally {
      compileBtn.disabled = false;
    }
  });
});
