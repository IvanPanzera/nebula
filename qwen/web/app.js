'use strict';
const $ = id => document.getElementById(id);
const icon = name => { const svg=document.createElementNS('http://www.w3.org/2000/svg','svg'); const use=document.createElementNS(svg.namespaceURI,'use'); use.setAttribute('href','#i-'+name); svg.append(use); return svg; };
const el = (tag, cls, text) => { const node=document.createElement(tag); if(cls)node.className=cls; if(text!==undefined)node.textContent=text; return node; };
const STORE='qwen.local.chats.v1';
let chats=[], current=null, busy=false, attachment=null, server={phase:'connecting',busy:false}, started=0, savedAt=0;
let toastTimer, restoredDraft='', thinking=false, selectedModel='flash-next', modelListSignature='';
let documentFile=null, documentMessage='';
try {
  const saved=JSON.parse(localStorage.getItem(STORE)||'{}');
  chats=Array.isArray(saved.chats)?saved.chats.filter(c=>typeof c.id==='string'&&typeof c.title==='string'&&Array.isArray(c.messages)):[];
  current=chats.find(c=>c.id===saved.current)?.id||null;
  for(const c of chats){
    const last=c.messages.at(-1);
    if(last?.pending){
      if(last.content){last.pending=false;last.cancelled=true;}
      else {c.messages.pop(); const user=c.messages.pop(); if(c.id===current)restoredDraft=user?.content||'';}
    }
  }
  if(['256','512','1024','2048'].includes(saved.maxTokens))$('max-tokens').value=saved.maxTokens;
  thinking=saved.thinking===true;
  selectedModel=typeof saved.model==='string'?saved.model:'flash-next';
} catch { /* An unavailable or invalid local store must not prevent chatting. */ }
const active = () => chats.find(c=>c.id===current);
function toast(text){$('toast').textContent=text;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,6500);}
function save(){try{localStorage.setItem(STORE,JSON.stringify({chats,current,thinking,model:selectedModel,maxTokens:$('max-tokens').value}));}catch{toast('La memoria del browser è piena o non disponibile. Esporta le chat che vuoi conservare.');}}
function modelInfo(){return server.models?.find(m=>m.id===selectedModel)||{id:'flash-next',label:'Flash-Next · speculativo',context:24576,speculative:true,available:true};}
function showModels(){
  const list=server.models;if(!list)return;
  const signature=JSON.stringify(list);
  if(signature!==modelListSignature){$('model-select').replaceChildren(...list.map(m=>{const option=el('option','',m.label+(m.available?'':' · non disponibile'));option.value=m.id;option.disabled=!m.available;return option;}));modelListSignature=signature;}
  if(!list.some(m=>m.id===selectedModel&&m.available))selectedModel='flash-next';
  $('model-select').value=selectedModel;
  const model=modelInfo();$('profile-model').textContent=model.label;
  $('model-note').textContent=(model.speculative?'MTP adattivo 4–16':'Modello autonomo su GPU')+` · contesto totale ${model.context.toLocaleString('it-IT')} token.`;
}
function showThinking(){const button=$('thinking-toggle');button.setAttribute('aria-checked',String(thinking));button.querySelector('.thinking-state').textContent=thinking?'On':'Off';}
function closeSidebar(){$('sidebar').classList.remove('open');$('scrim').hidden=true;}
function selectChat(id){if(busy)return;current=id;if(active())selectedModel=active().model||'flash-next';showModels();attachment=null;showAttachment();$('prompt').value='';resize();render();showStatus();save();closeSidebar();}
function renderHistory(){
  const nav=$('history');nav.replaceChildren();$('chat-count').textContent=chats.length;
  if(!chats.length){nav.append(el('p','empty-history','Le tue conversazioni appariranno qui. Un’idea alla volta.'));return;}
  for(const chat of chats){
    const row=el('div','history-item'+(chat.id===current?' active':''));
    const button=el('button');button.append(icon('chat'),el('span','',chat.title));button.title=chat.title;button.disabled=busy;button.onclick=()=>selectChat(chat.id);
    if(chat.id===current)button.setAttribute('aria-current','page');
    const remove=el('button','delete-chat');remove.append(icon('trash'));remove.title='Elimina conversazione';remove.setAttribute('aria-label','Elimina '+chat.title);remove.disabled=busy;
    remove.onclick=()=>{if(!confirm('Eliminare questa conversazione dal browser?'))return;chats=chats.filter(c=>c.id!==chat.id);if(current===chat.id)current=null;render();save();};
    row.append(button,remove);nav.append(row);
  }
}

// Construct nodes from text, never HTML from the model. This small renderer
// supports the chat essentials without a CDN or executable model markup.
function inline(parent,text){
  const pattern=/(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/g;
  let end=0;
  for(const match of text.matchAll(pattern)){
    parent.append(document.createTextNode(text.slice(end,match.index)));
    const value=match[0];let node;
    if(value.startsWith('`'))node=el('code','',value.slice(1,-1));
    else if(value.startsWith('**'))node=el('strong','',value.slice(2,-2));
    else {const parts=value.match(/^\[([^\]]+)\]\(([^)]+)\)$/);node=el('a','',parts[1]);node.href=parts[2];node.target='_blank';node.rel='noopener noreferrer';}
    parent.append(node);end=match.index+value.length;
  }
  parent.append(document.createTextNode(text.slice(end)));
}
async function copy(text,button){try{await navigator.clipboard.writeText(text);if(button){button.replaceChildren(icon('check'));setTimeout(()=>button.replaceChildren(icon('copy')),1500);}else toast('Copiato negli appunti.');}catch{toast('Copia non disponibile: seleziona il testo e usa Ctrl+C.');}}
function markdown(parent,text){
  parent.replaceChildren();const lines=text.split('\n');let i=0;
  while(i<lines.length){
    const line=lines[i];
    if(!line.trim()){i++;continue;}
    if(/^\s*```/.test(line)){
      const language=line.replace(/^\s*```/,'').trim();const code=[];i++;
      while(i<lines.length&&!/^\s*```/.test(lines[i]))code.push(lines[i++]);if(i<lines.length)i++;
      const block=el('div','code-block'),head=el('div','code-header'),button=el('button','', 'Copia');button.prepend(icon('copy'));button.setAttribute('aria-label','Copia codice');button.onclick=()=>copy(code.join('\n'));
      head.append(el('span','',language||'codice'),button);const pre=el('pre');pre.append(el('code','',code.join('\n')));block.append(head,pre);parent.append(block);continue;
    }
    const heading=line.match(/^(#{1,4})\s+(.+)/);
    if(heading){const h=el('h'+Math.min(4,heading[1].length+1));inline(h,heading[2]);parent.append(h);i++;continue;}
    if(/^\s*([-*+] |\d+\. )/.test(line)){
      const ordered=/^\s*\d+\. /.test(line),list=el(ordered?'ol':'ul');
      const match=ordered?/^\s*\d+\. (.*)/:/^\s*[-*+] (.*)/;
      while(i<lines.length&&match.test(lines[i])){const item=el('li');inline(item,lines[i++].match(match)[1]);list.append(item);}parent.append(list);continue;
    }
    if(line.startsWith('> ')){const quote=el('blockquote');inline(quote,line.slice(2));parent.append(quote);i++;continue;}
    const paragraph=[line];i++;
    while(i<lines.length&&lines[i].trim()&&!/^(\s*```|#{1,4} |\s*[-*+] |\s*\d+\. |> )/.test(lines[i]))paragraph.push(lines[i++]);
    const p=el('p');inline(p,paragraph.join('\n'));parent.append(p);
  }
}
function fillAssistant(article,message){
  const content=article.querySelector('.message-content');
  if(message.content)markdown(content,message.content);
  else if(message.pending){const dots=el('div','thinking-dots');dots.setAttribute('aria-label','Qwen sta preparando la risposta');dots.append(el('span'),el('span'),el('span'));content.replaceChildren(dots);}
  else {const text=message.metrics?.thinking&&!message.metrics.reasoning_complete?(message.cancelled?'Ragionamento interrotto prima della risposta.':'Il limite di token è stato raggiunto durante il ragionamento. Aumenta la lunghezza nelle impostazioni oppure disattiva Thinking.'):'Nessun testo di risposta generato.';content.replaceChildren(el('p','empty-answer',text));}
  const meta=article.querySelector('.message-meta');meta.replaceChildren();
  if(message.pending)return;
  if(message.metrics){
    const m=message.metrics,grid=el('div','metrics-grid');grid.setAttribute('aria-label','Statistiche della risposta');
    const number=(value,digits=0)=>Number.isFinite(value)?value.toLocaleString('it-IT',{maximumFractionDigits:digits,minimumFractionDigits:digits}):'—';
    const metric=(label,value,unit,title)=>{const card=el('div','metric');card.title=title;card.append(el('span','metric-label',label));const row=el('div','metric-value',value);row.append(el('span','metric-unit',unit));card.append(row);grid.append(card);return card;};
    const total=metric('Token risposta',number(m.answer_tokens??m.output_tokens),'token','Token effettivamente generati per la risposta finale. Esclude il ragionamento nascosto, il prompt e le proposte draft scartate.');
    total.append(el('span','metric-detail',`Generati complessivi: ${number(m.output_tokens)}`));
    if(m.thinking)total.append(el('span','metric-detail',`Thinking On · ${number(m.reasoning_tokens)} token di ragionamento`));
    metric('Velocità prefill',number(m.prefill_tokens_per_second,1),'token/s','Token nuovi del prompt elaborati, divisi per il tempo di prefill. Il prefisso riutilizzato non viene contato di nuovo.');
    metric('Velocità generazione',number(m.tokens_per_second,2),'token/s','Velocità di generazione misurata dal motore. Esclude caricamento e prefill; include il ragionamento se attivo.');
    if(m.speculative!==false){
    metric('Draft accettati · GPU',number(m.draft_output_tokens),'token','Token generati dal MTP e confermati dal target GPU, incluso il ragionamento se attivo. Esclude EOS e proposte scartate.');
    metric('Target · GPU',number(m.target_output_tokens),'token','Token generati direttamente dal target, inclusi ancoraggi, correzioni e ragionamento se attivo.');
    metric('Target · CPU',number(m.target_cpu_tokens),'token','Il motore attuale esegue tutto il calcolo del target in GPU. La CPU gestisce caricamento, trasferimenti e controllo.');
    const chain=metric('Catena draft max',number(m.max_accepted_draft),'token','Massimo numero di token draft consecutivi accettati ed emessi in un singolo blocco.');
    chain.classList.add('chain-metric');
    chain.append(el('span','metric-detail',`N massimo proposto: ${number(m.max_draft_used)}`));
    if(m.draft_policy==='adaptive')chain.append(el('span','metric-detail',`N adattivo: ${number(m.draft_min)}–${number(m.draft_max)}`));
    }else{
      metric('Generati · GPU',number(m.target_output_tokens),'token','Token prodotti dal modello autonomo, incluso il ragionamento se attivo.');
    }
    if(m.speculative!==false&&Number.isFinite(m.expert_handoff_events)){
      const card=metric('Expert handoff',number(m.expert_handoff_events),'passaggi nei layer','Esecuzioni di un layer del target che hanno richiesto almeno un esperto dalla RAM. Un batch conta una volta per layer, anche con più esperti mancanti. Include prefill, verifica e replay.');
      card.classList.add('handoff-metric');
      const fraction=m.layer_executions?100*m.expert_handoff_events/m.layer_executions:0;
      card.append(el('span','handoff-summary',`${number(fraction,1)}% dei passaggi · Prefill ${number(m.prefill_handoff_events)} · Generazione ${number(m.decode_handoff_events)}`));
      card.append(el('span','metric-detail',`${number(m.expert_uploads)} esperti trasferiti · ${number(m.expert_upload_gib,2)} GiB dalla RAM`));
    }
    meta.append(grid);
    meta.append(el('p','metrics-note',(m.speculative===false?'Modello autonomo su GPU · senza draft.':'Il target calcola in GPU; gli esperti mancanti vengono caricati dalla RAM.')+(m.thinking?' Velocità e contatori includono il ragionamento nascosto.':'')));
    if(m.speculative!==false&&m.draft_trace?.length){
      const details=el('details','handoff-details');details.append(el('summary','','Progressione di N · dettaglio dei blocchi draft'));
      const table=el('table'),head=el('thead'),hr=el('tr');for(const title of ['Blocco','N richiesto','Accettati / proposti','N successivo'])hr.append(el('th','',title));head.append(hr);table.append(head);
      const body=el('tbody');m.draft_trace.forEach(([n,proposed,accepted,next],i)=>{const row=el('tr');for(const value of [i+1,n,`${number(accepted)} / ${number(proposed)}`,next])row.append(el('td','',String(value)));body.append(row);});table.append(body);
      const wrap=el('div','handoff-table');wrap.append(table);details.append(wrap);
      details.append(el('p','metrics-note','Le proposte effettive possono essere inferiori a N vicino alla fine della risposta, al limite del contesto o al token di fine testo.'));
      meta.append(details);
    }
    if(m.speculative!==false&&Array.isArray(m.handoffs_by_layer)){
      const details=el('details','handoff-details');details.append(el('summary','','Expert handoff · dettaglio dei 48 layer'));
      const table=el('table'),head=el('thead'),hr=el('tr');for(const title of ['Layer','Prefill','Generazione','Totale / passaggi'])hr.append(el('th','',title));head.append(hr);table.append(head);
      const body=el('tbody');for(const layer of m.handoffs_by_layer){const row=el('tr');for(const value of [layer.layer,number(layer.prefill),number(layer.decode),`${number(layer.events)} / ${number(layer.calls)}`])row.append(el('td','',String(value)));body.append(row);}table.append(body);const wrap=el('div','handoff-table');wrap.append(table);details.append(wrap);meta.append(details);
    }
  }
  const actions=el('div','message-actions');
  if(message.content){const button=el('button','icon-button');button.append(icon('copy'));button.setAttribute('aria-label','Copia risposta');button.title='Copia risposta';button.onclick=()=>copy(message.content,button);actions.append(button);}
  if(message.metrics){const m=message.metrics;actions.append(el('span','',`${m.output_tokens} token generati totali`),el('span','',`${Number(m.prefill_seconds).toFixed(1).replace('.',',')} s di prefill`));if(m.reused_prefix_tokens>0)actions.append(el('span','',`${m.reused_prefix_tokens.toLocaleString('it-IT')} token riutilizzati`));if(m.at_limit)actions.append(el('span','limit',m.stop_reason==='context_limit'?'Contesto esaurito':'Limite di risposta raggiunto'));}
  if(message.cancelled)actions.append(el('span','limit','Risposta interrotta'));
  meta.append(actions);
}
function renderMessages(){
  const chat=active(),container=$('messages');container.replaceChildren();$('welcome').hidden=!!chat?.messages.length;
  for(const message of chat?.messages||[]){
    const article=el('article','message '+message.role);
    if(message.role==='user'){article.append(el('div','message-content'+(message.content.length>1000?' long':''),message.display||message.content));}
    else {const head=el('div','message-head'),mark=el('span','mini-mark');mark.append(icon('spark'));head.append(mark,el('span','',message.metrics?.model||server.models?.find(m=>m.id===message.model)?.label||'Qwen'));article.append(head,el('div','message-content'),el('div','message-meta'));fillAssistant(article,message);}
    container.append(article);
  }
  const metric=[...(chat?.messages||[])].reverse().find(m=>m.metrics)?.metrics;
  $('context-label').textContent=metric?`${metric.context_tokens.toLocaleString('it-IT')} / ${(metric.context||modelInfo().context).toLocaleString('it-IT')} token · Verifica le risposte importanti`:`${modelInfo().context.toLocaleString('it-IT')} token di contesto · Verifica le risposte importanti`;
  $('export').disabled=!chat?.messages.length;
}
function scrollBottom(force=false){const area=$('scroll-area');if(force||area.scrollHeight-area.scrollTop-area.clientHeight<160)area.scrollTop=area.scrollHeight;}
function render(){renderHistory();renderMessages();controls();if(active()?.messages.length)scrollBottom(true);else $('scroll-area').scrollTop=0;}
function resize(){$('prompt').style.height='auto';$('prompt').style.height=Math.min($('prompt').scrollHeight,174)+'px';controls();}
function showAttachment(){$('attachment').hidden=!attachment;$('attachment-name').textContent=attachment?.name||'';controls();}
function controls(){
  const working=busy||server.busy;
  $('send').hidden=working;$('stop').hidden=!working;
  $('send').disabled=!$('prompt').value.trim()||working||server.external_busy||server.phase==='offline'||server.phase==='connecting';
  $('prompt').disabled=working;$('attach').disabled=working;$('new-chat').disabled=busy;$('max-tokens').disabled=busy;
  $('thinking-toggle').disabled=working;
  $('model-select').disabled=working||server.phase==='offline'||server.phase==='connecting';
  $('stop').disabled=!!server.cancel_requested;
  $('document-convert').disabled=working||server.external_busy||!server.documents_available;
  for(const id of ['document-close','document-pages','document-force'])$(id).disabled=working;
  $('activity').hidden=!working;
}
const labels={idle:'Da avviare',ready:'Modello pronto',preparing:'Preparazione',loading:'Caricamento',prefill:'Legge il contesto',thinking:'Sta ragionando',generating:'Sta rispondendo',document:'Legge il documento',ocr:'Riconosce il documento',error:'Da verificare',offline:'Non connesso',connecting:'Connessione…'};
function showStatus(){
  const model=modelInfo(),working=busy||server.busy;
  const phase=!working&&server.model_id&&server.model_id!==selectedModel&&server.phase!=='offline'?'idle':server.phase;
  const label=server.external_busy?'Chat terminale aperta':labels[phase]||'Connessione…';
  $('status').querySelector('span').textContent=label;
  $('status').className='status'+(phase==='ready'?' ready':working?' busy':phase==='error'||phase==='offline'?' error':'');
  let notice='';
  if(server.external_busy)notice='La chat da terminale è aperta. Chiudila con /exit: potrai poi usare Qwen da questa pagina.';
  else if(phase==='loading')notice='Caricamento del modello selezionato. Lascia aperto il terminale; le risposte successive useranno il modello già in memoria.';
  else if(phase==='idle')notice=model.speculative?'Flash-Next si carica al primo messaggio; il trasferimento dei pesi in RAM può richiedere diversi minuti.':'Il modello light si carica al primo messaggio. Il motore precedente verrà prima scaricato dalla memoria.';
  else if(phase==='offline')notice='Il server non è raggiungibile. Avvia avvia_qwen_webui.bat e lascia aperto il terminale.';
  $('notice').hidden=!notice;$('notice').textContent=notice;
  let text=phase==='loading'?'Qwen si sta preparando'+(server.load_progress!==undefined?` · pesi in memoria ${server.load_progress}%`:'…'):phase==='prefill'?'Qwen sta leggendo la conversazione…':phase==='generating'?'Qwen sta scrivendo…':'Preparazione della risposta…';
  if(phase==='thinking')text='Qwen sta ragionando · il ragionamento rimane nascosto…';
  if(phase==='ocr'||phase==='document')text=documentMessage||'Conversione del documento in Markdown…';
  if(server.cancel_requested)text='Interruzione richiesta · attendo la fine del calcolo in corso…';
  $('activity-text').textContent=text;controls();
}
async function poll(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error();server=await r.json();showModels();}catch{server={phase:'offline',busy:false};}showStatus();}
async function submit(event){
  event.preventDefault();if($('send').disabled||busy)return;
  const typed=$('prompt').value.trim(),file=attachment;
  const content=file?`${typed}\n\n--- Documento: ${file.name} ---\n${file.text}\n--- Fine documento ---`:typed;
  let chat=active();if(!chat){chat={id:crypto.randomUUID(),title:typed.slice(0,65),messages:[],created:Date.now()};chats.unshift(chat);current=chat.id;}chat.model=selectedModel;
  const user={role:'user',content,display:file?`${typed}\n\n📄 ${file.name}`:typed};
  const answer={role:'assistant',content:'',pending:true,model:selectedModel};chat.messages.push(user,answer);
  busy=true;started=Date.now();attachment=null;$('prompt').value='';showAttachment();resize();render();save();
  let complete=false;
  try{
    const response=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json','X-Qwen-Client':'webui'},body:JSON.stringify({model:selectedModel,messages:chat.messages.slice(0,-1).map(m=>({role:m.role,content:m.content,...(typeof m.reasoning_content==='string'&&(m.model||'flash-next')===selectedModel?{reasoning_content:m.reasoning_content}:{})})),max_tokens:Number($('max-tokens').value),thinking})});
    if(!response.ok){const error=await response.json();throw Error(error.error||'Impossibile iniziare la risposta.');}
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
    while(true){const chunk=await reader.read();buffer+=decoder.decode(chunk.value||new Uint8Array(),{stream:!chunk.done});let pos;
      while((pos=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,pos);buffer=buffer.slice(pos+1);if(!line.trim())continue;const data=JSON.parse(line);
        if(data.type==='error')throw Error(data.message);
        if(data.type==='phase'){server={...server,phase:data.phase,busy:true};showStatus();}
        if(data.type==='token'||data.type==='done'){
          const area=$('scroll-area'),stick=area.scrollHeight-area.scrollTop-area.clientHeight<160;
          if(data.type==='token')answer.content+=data.text;
          else{answer.content=data.text;answer.pending=false;answer.metrics=data.metrics;answer.cancelled=data.cancelled;answer.reasoning_content=data.reasoning_content;complete=true;}
          fillAssistant($('messages').lastElementChild,answer);if(stick)scrollBottom(true);
          if(Date.now()-savedAt>3000){save();savedAt=Date.now();}
        }
      }
      if(chunk.done)break;
    }
    if(!complete)throw Error('La connessione si è interrotta prima della fine della risposta.');
    if(!answer.content&&!answer.metrics?.output_tokens){chat.messages.splice(-2);$('prompt').value=typed;attachment=file;}
  }catch(error){
    if(answer.content){answer.pending=false;answer.cancelled=true;}
    else{chat.messages.splice(-2);$('prompt').value=typed;attachment=file;}
    toast(error.message||'Errore di connessione.');
  }finally{
    busy=false;server.busy=false;showAttachment();resize();render();save();await poll();if(!$('prompt').disabled)$('prompt').focus();
  }
}
$('composer').addEventListener('submit',submit);
$('prompt').addEventListener('input',resize);
$('prompt').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();$('composer').requestSubmit();}});
$('new-chat').onclick=()=>selectChat(null);
document.querySelectorAll('[data-prompt]').forEach(button=>button.onclick=()=>{$('prompt').value=button.dataset.prompt;resize();$('prompt').focus();});
$('menu').onclick=()=>{$('sidebar').classList.add('open');$('scrim').hidden=false;};$('scrim').onclick=closeSidebar;
$('settings-toggle').onclick=()=>{$('settings').hidden=!$('settings').hidden;$('settings-toggle').setAttribute('aria-expanded',String(!$('settings').hidden));};
document.addEventListener('keydown',e=>{if(e.key==='Escape'){closeSidebar();$('settings').hidden=true;$('settings-toggle').setAttribute('aria-expanded','false');}});
$('max-tokens').onchange=save;
$('model-select').onchange=()=>{selectedModel=$('model-select').value;if(active())active().model=selectedModel;showModels();showStatus();renderMessages();save();};
$('thinking-toggle').onclick=()=>{thinking=!thinking;showThinking();save();};
$('attach').onclick=()=>$('file').click();
$('file').onchange=async()=>{
  const file=$('file').files[0];$('file').value='';if(!file)return;
  if(/\.(pdf|png|jpe?g|webp)$/i.test(file.name)){
    if(file.size>25*1024*1024){toast('Scegli un documento più piccolo di 25 MiB.');return;}
    documentFile=file;$('document-name').textContent=file.name;$('document-pages').value='';
    $('document-force').checked=false;$('document-panel').hidden=false;$('document-result').hidden=true;
    if(!server.documents_available)toast('Il modulo OCR non è ancora disponibile in questa installazione.');
    controls();return;
  }
  if(file.size>256000){toast('Scegli un file di testo più piccolo di 256 KB. Il contenuto deve rientrare nel contesto del modello.');return;}
  try{const text=await file.text();if(text.includes('\0'))throw Error('Il file deve contenere testo.');attachment={name:file.name,text};showAttachment();$('prompt').focus();}catch(error){toast(error.message);}
};
$('document-close').onclick=()=>{documentFile=null;$('document-panel').hidden=true;};
$('document-convert').onclick=async()=>{
  if(!documentFile||busy||$('document-convert').disabled)return;
  busy=true;started=Date.now();documentMessage='Carica il documento…';server.phase='document';showStatus();
  let complete=false;
  try{
    const response=await fetch('/api/documents',{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Qwen-Client':'webui','X-Document-Name':encodeURIComponent(documentFile.name),'X-Document-Pages':$('document-pages').value,'X-Document-OCR':String($('document-force').checked)},body:documentFile});
    if(!response.ok){const result=await response.json();throw Error(result.error||'Conversione non disponibile.');}
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
    while(true){
      const chunk=await reader.read();buffer+=decoder.decode(chunk.value||new Uint8Array(),{stream:!chunk.done});let pos;
      while((pos=buffer.indexOf('\n'))>=0){
        const line=buffer.slice(0,pos);buffer=buffer.slice(pos+1);if(!line.trim())continue;
        const data=JSON.parse(line);
        if(data.type==='error')throw Error(data.message);
        if(data.type==='phase'){server.phase=data.phase;documentMessage=data.message||'';showStatus();}
        if(data.type==='document_cancelled'){complete=true;toast('Conversione interrotta.');}
        if(data.type==='document_done'){
          complete=true;$('document-panel').hidden=true;$('document-result').hidden=false;
          const ocrPages=data.page_details.filter(p=>p.method==='PaddleOCR-VL-1.5').length;
          $('document-summary').textContent=`${data.pages} pagine · ${ocrPages} con OCR · ${data.seconds.toFixed(1)} s`;
          $('document-md').href=`/api/documents/${data.id}/document.md`;$('document-md').download=data.name;
          $('document-zip').href=`/api/documents/${data.id}/document.zip`;$('document-zip').download=data.name.replace(/\.md$/,'.zip');
          $('document-preview').textContent=data.attachment_allowed?data.text:'Documento troppo lungo per un singolo allegato. Scarica il Markdown oppure converti un intervallo di pagine più piccolo.';
          if(data.attachment_allowed){
            attachment={name:data.name,text:data.text};
            if(server.models?.some(m=>m.id==='light'&&m.available)){selectedModel='light';showModels();}
            showAttachment();toast('Markdown pronto e allegato. Scrivi cosa vuoi sapere sul documento.');
          }else toast('Conversione completata. Per la chat scegli meno pagine; il file completo è scaricabile.');
        }
      }
      if(chunk.done)break;
    }
    if(!complete)throw Error('Connessione interrotta durante la conversione.');
  }catch(error){toast(error.message||'Conversione non riuscita.');}
  finally{busy=false;server.busy=false;documentMessage='';await poll();save();$('prompt').focus();}
};
$('remove-attachment').onclick=()=>{attachment=null;showAttachment();};
$('stop').onclick=async()=>{try{const r=await fetch('/api/stop',{method:'POST',headers:{'X-Qwen-Client':'webui'}});if(!r.ok)throw Error();server.cancel_requested=true;showStatus();}catch{toast('Impossibile richiedere l’interruzione. Controlla il terminale.');}};
$('export').onclick=()=>{const chat=active();if(!chat)return;const text=`# ${chat.title}\n\n`+chat.messages.map(m=>`## ${m.role==='user'?'Tu':'Qwen'}\n\n${m.content}`).join('\n\n');const url=URL.createObjectURL(new Blob([text],{type:'text/markdown;charset=utf-8'}));const link=el('a');link.href=url;link.download='qwen-conversazione.md';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
setInterval(()=>{if(busy){const s=Math.floor((Date.now()-started)/1000);$('activity-time').textContent=s<60?`${s}s`:`${Math.floor(s/60)}m ${s%60}s`;}else $('activity-time').textContent='';},1000);
$('prompt').value=restoredDraft;showThinking();render();resize();poll();setInterval(poll,2000);
