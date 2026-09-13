const API_BASE = "http://localhost:8000";
const $ = id => document.getElementById(id);
let mode = "text", pollTimer, started;

document.querySelectorAll(".tab").forEach(btn => btn.onclick = () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
  btn.classList.add("active");
  mode = btn.dataset.mode;
  $("imageBox").classList.toggle("hidden", mode !== "image");
});

document.querySelectorAll("[data-prompt]").forEach(btn => btn.onclick = () => $("prompt").value = btn.dataset.prompt);
$("refresh").onclick = loadVideos;

function elapsed(){
  const s = Math.floor((Date.now()-started)/1000);
  $("timer").textContent = String(Math.floor(s/60)).padStart(2,"0")+":"+String(s%60).padStart(2,"0");
}

$("generate").onclick = async () => {
  const prompt = $("prompt").value.trim();
  if(!prompt) return alert("Fadlan geli prompt.");
  $("progress").classList.remove("hidden");
  $("result").classList.add("hidden");
  $("status").textContent = "Generating…";
  started = Date.now();
  clearInterval(pollTimer); pollTimer = setInterval(elapsed, 500);

  try{
    let r;
    if(mode === "text"){
      r = await fetch(API_BASE+"/api/generate", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({
          prompt, language:$("language").value,
          aspect_ratio:$("ratio").value, duration:Number($("duration").value),
          quality:$("quality").value, speed:$("speed").value, quality:$("quality").value
        })
      });
    }else{
      const fd = new FormData();
      fd.append("prompt",prompt); fd.append("language",$("language").value);
      fd.append("aspect_ratio",$("ratio").value); fd.append("duration",$("duration").value);
      fd.append("quality",$("quality").value); fd.append("speed",$("speed").value); fd.append("quality",$("quality").value);
      fd.append("image",$("image").files[0]);
      r = await fetch(API_BASE+"/api/generate-image",{method:"POST",body:fd});
    }
    const data = await r.json();
    if(!r.ok) throw new Error(data.detail || "Generation failed");
    await poll(data.id);
  }catch(e){
    $("status").textContent = e.message;
  }
};

async function poll(id){
  let done=false;
  while(!done){
    const r=await fetch(API_BASE+"/api/predictions/"+id);
    const d=await r.json();
    $("status").textContent = d.status || "processing…";
    if(["succeeded","failed","canceled"].includes(d.status)){
      done=true;
      clearInterval(pollTimer);
      if(d.status==="succeeded" && d.output){
        const url=Array.isArray(d.output)?d.output[0]:d.output;
        $("result").innerHTML=`<video controls src="${url}"></video><br><a class="download" href="${url}" target="_blank" download>⬇ Download</a>`;
        $("result").classList.remove("hidden");
      }else $("status").textContent = "Video generation failed.";
      loadVideos();
    }else await new Promise(x=>setTimeout(x,2500));
  }
}

async function loadVideos(){
  try{
    const r=await fetch(API_BASE+"/api/videos"); const items=await r.json();
    $("videos").innerHTML = items.map(v => {
      const out=Array.isArray(v.output)?v.output[0]:v.output;
      return `<div class="video-item">${out?`<video controls src="${out}"></video>`:"<div>⏳ "+(v.status||"starting")+"</div>"}<small>${new Date(v.created_at*1000).toLocaleString()}</small><p>${escapeHtml(v.prompt)}</p></div>`;
    }).join("") || "<p>No videos yet.</p>";
  }catch(e){ $("videos").innerHTML="<p>Backend not connected.</p>"; }
}
function escapeHtml(s){return s.replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]))}
loadVideos();
