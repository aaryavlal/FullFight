
document.addEventListener('DOMContentLoaded', () => {
  let animeData = {};
  let uploadedFiles = [];


  document.getElementById('fightForm').addEventListener('submit', async e => {
    e.preventDefault();
    const query = document.getElementById('query').value.trim();
    if (!query) return alert('Please enter a fight query');
    try {
      const res = await fetch('/parse_fight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      if (!res.ok) throw new Error(`Status ${res.status}`);
      animeData = await res.json();
      document.getElementById('parsedOutput').innerText =
        `Anime: ${animeData.anime}, Fighters: ${animeData.fighters.join(' vs ')}`;
    } catch (err) {
      console.error(err);
      alert('Error parsing fight: ' + err.message);
    }
  });


  document.getElementById('uploadForm').addEventListener('submit', async e => {
    e.preventDefault();
    const formData = new FormData(e.target);
    try {
      const res = await fetch('/upload_episodes', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error(`Status ${res.status}`);
      const data = await res.json();
      uploadedFiles = data.saved_files.map(path => path.split('/').pop());
      document.getElementById('uploadStatus').innerText =
        `Uploaded ${uploadedFiles.length} episode(s).`;
      document.getElementById('compile-btn').disabled = false;
    } catch (err) {
      console.error(err);
      alert('Error uploading files: ' + err.message);
    }
  });

  
  document.getElementById('compile-btn').addEventListener('click', async () => {
    if (!uploadedFiles.length) {
      return alert('Please upload episodes first.');
    }
    try {
 
      const tsRes = await fetch('/get_timestamps', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(animeData)
      });
      if (!tsRes.ok) throw new Error(`Status ${tsRes.status}`);
      const { timestamps } = await tsRes.json();

      const compileRes = await fetch('/compile_fight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          timestamps,
          video_files: uploadedFiles
        })
      });
      if (!compileRes.ok) throw new Error(`Status ${compileRes.status}`);
      const { output_file } = await compileRes.json();
      const videoUrl = `/output/${output_file}`;

      document.getElementById('result').innerHTML = `
        <h3>✅ Your fight is ready:</h3>
        <video controls width="600" src="${videoUrl}"></video><br>
        <a href="${videoUrl}" download>Download Video</a>
      `;
    } catch (err) {
      console.error(err);
      alert('Error compiling fight: ' + err.message);
    }
  });
});
