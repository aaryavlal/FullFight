document.addEventListener('DOMContentLoaded', () => {
  let uploadedFiles = [];

  const form         = document.getElementById('uploadForm');
  const uploadStatus = document.getElementById('uploadStatus');
  const compileBtn   = document.getElementById('compile-btn');
  const resultDiv    = document.getElementById('result');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    uploadStatus.textContent = 'Uploading...';
    compileBtn.disabled = true;
    resultDiv.innerHTML = '';

    const data = new FormData(form);

    try {
      const res  = await fetch(form.action, { method: 'POST', body: data });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || 'Upload failed');

      uploadedFiles = json.saved_files || [json.filename];
      uploadStatus.textContent = ` Uploaded ${uploadedFiles.length} file(s).`;
      compileBtn.disabled = false;
    } catch (err) {
      console.error(err);
      uploadStatus.textContent = ` ${err.message}`;
      alert('Upload error: ' + err.message);
    }
  });


  compileBtn.addEventListener('click', async () => {
    if (!uploadedFiles.length) {
      return alert('Please upload episodes first.');
    }

    compileBtn.disabled = true;
    resultDiv.innerHTML = 'Compiling…';

    try {
      const compileRes = await fetch('/compile_fight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: uploadedFiles[0] })
      });

      if (!compileRes.ok) throw new Error(`Compile error: ${compileRes.status}`);
      const { output_file } = await compileRes.json();

      const videoUrl = `/output/${output_file}`;
      resultDiv.innerHTML = `
        <h3>Your fight is ready:</h3>
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
